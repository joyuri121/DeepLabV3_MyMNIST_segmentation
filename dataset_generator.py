"""
MNIST 数字を貼り付けた合成スペクトログラム風データセット生成モジュール。

本モジュールは、ERG (あらせ衛星) の OFA-SPEC データに対するセグメンテーション課題と
同型の問題設定を、公開データセット MNIST のみを用いて構築するためのものである。
研究室内の元タスク (ヒス・コーラスの semantic segmentation) と
本質的に同じ構造 (縦長キャンバス・疎な前景・ノイズ背景・multi-class) を保ちつつ、
データポリシー上の制約を完全に回避する目的で設計している。

画像構造:
    - サイズ: 1024 (H) x 180 (W) の縦長キャンバス
    - 内容: MNIST 数字 (0.5〜3.0 倍スケール) + 1/f ノイズ + 縦ノイズ
            + 横バンドノイズ + ベジエ曲線
    - 数字同士の重なりは許可する (multi-hot ラベルでピクセル単位の多重所属を表現する)

正解ラベル構造:
    - 形状: (11, H, W) の uint8 配列
    - チャネル 0: 背景 (どの数字にも属さないピクセルが 1)
    - チャネル 1〜10: 数字 0〜9 のセグメンテーションマスク
    - 重なり部のピクセルは複数チャネルで同時に 1 となる (multi-hot)

設計判断 (案 A):
    画像生成とマスク生成は同じ関数 generate_canvas_and_mask 内で同時に行う。
    これは数字配置の乱数列を共有しないと両者の整合性が崩れるためであり、
    関数を分割するとバグ混入リスクが高いと判断したからである。
    ユーザは Jupyter Notebook から build_dataset を呼ぶことで、
    画像とマスクのペアを一括生成・保存する。
"""

import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets


# ==================== 定数 ====================
CANVAS_HEIGHT = 1024
CANVAS_WIDTH = 180
NUM_DIGIT_CLASSES = 10
NUM_CLASSES_WITH_BACKGROUND = 11  # 背景 + 数字 0〜9
DIGIT_PIXEL_THRESHOLD = 50        # この輝度を超える MNIST ピクセルを「数字本体」と見なす

# 生成パラメータのデフォルト値
DEFAULT_DIGITS_PER_IMAGE_RANGE = (3, 10)
DEFAULT_SCALE_RANGE = (3.0, 3.0)  # 3.0 倍固定 (元は (0.5, 3.0) の連続一様)
DEFAULT_BEZIER_COUNT_RANGE = (0, 3)
DEFAULT_VERTICAL_STREAK_COUNT_RANGE = (0, 3)
DEFAULT_HORIZONTAL_BAND_COUNT_RANGE = (0, 3)


# ==================== MNIST 取得 ====================
def load_mnist_arrays(train=True, mnist_root='./mnist_data'):
    """torchvision 経由で MNIST を取得して numpy 配列で返す。

    Args:
        train: True なら訓練分 (60000枚)、False ならテスト分 (10000枚) を取得する。
        mnist_root: MNIST のダウンロード先ディレクトリ。

    Returns:
        images: (N, 28, 28) uint8 の数字画像
        labels: (N,) int の数字ラベル (0〜9)
    """
    mnist = datasets.MNIST(root=mnist_root, train=train, download=True)
    images = mnist.data.numpy().astype(np.uint8)
    labels = mnist.targets.numpy().astype(np.int64)
    return images, labels


# ==================== ノイズ生成 ====================
def _generate_one_over_f_noise(shape, rng, amplitude=30.0):
    """1/f ノイズを生成する。

    実スペクトログラムの背景に乗る機器熱雑音を模擬する目的である。
    周波数領域で振幅スペクトルを 1/f 形に整形し、ランダム位相を掛けて逆FFTで空間領域に戻す。

    Args:
        shape: (H, W) の出力形状
        rng: numpy 乱数ジェネレータ
        amplitude: 出力の最大振幅

    Returns:
        (H, W) float32 のノイズ配列 (値域はおおむね [0, amplitude])
    """
    height, width = shape
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.fftfreq(width)[None, :]
    freq_magnitude = np.sqrt(freq_y ** 2 + freq_x ** 2)
    freq_magnitude[0, 0] = 1.0  # DC 成分の 0 除算を回避する

    spectrum_envelope = 1.0 / freq_magnitude
    random_complex = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    noise_in_freq_domain = spectrum_envelope * random_complex
    noise = np.real(np.fft.ifft2(noise_in_freq_domain))

    # 振幅を [0, amplitude] に正規化する
    noise -= noise.min()
    max_value = noise.max()
    if max_value > 0:
        noise *= amplitude / max_value
    return noise.astype(np.float32)


def _add_vertical_streaks(canvas, rng, count):
    """縦方向の短時間広帯域ノイズを canvas に in-place で加える。

    実スペクトログラムでの機器の再起動や振動雑音に相当する想定である。
    """
    _, width = canvas.shape
    for _ in range(count):
        center_x = int(rng.integers(0, width))
        streak_half_width = int(rng.integers(1, 4))
        intensity = float(rng.uniform(80, 180))
        x_start = max(0, center_x - streak_half_width)
        x_end = min(width, center_x + streak_half_width + 1)
        canvas[:, x_start:x_end] = np.maximum(
            canvas[:, x_start:x_end], intensity
        )


def _add_horizontal_bands(canvas, rng, count):
    """横方向のバンドノイズを canvas に in-place で加える。

    実スペクトログラムでの定常的な機器ノイズに相当する想定である。
    """
    height, _ = canvas.shape
    for _ in range(count):
        center_y = int(rng.integers(0, height))
        band_half_thickness = int(rng.integers(1, 4))
        intensity = float(rng.uniform(60, 140))
        y_start = max(0, center_y - band_half_thickness)
        y_end = min(height, center_y + band_half_thickness + 1)
        canvas[y_start:y_end, :] = np.maximum(
            canvas[y_start:y_end, :], intensity
        )


# ==================== ベジエ曲線描画 ====================
def _evaluate_bezier_at(control_points, t):
    """de Casteljau のアルゴリズムでベジエ曲線上の 1 点を評価する。

    Args:
        control_points: (N, 2) 制御点座標 (y, x 順)
        t: スカラー、媒介変数 [0, 1]

    Returns:
        (2,) 曲線上の点座標
    """
    points = control_points.copy()
    while len(points) > 1:
        points = (1.0 - t) * points[:-1] + t * points[1:]
    return points[0]


def _draw_bezier_curve(canvas, rng):
    """1 本のベジエ曲線 (制御点 3 or 4 個) を canvas に描画する。

    数字に似た滑らかなストロークだが数字ではない「紛らわしい背景物体」を作る目的である。
    モデルが形状の滑らかさだけで数字判定しないようにするための negative sample を提供する。
    """
    height, width = canvas.shape
    num_control_points = int(rng.integers(3, 5))  # 3 or 4 点
    control_points = np.column_stack([
        rng.uniform(0, height, size=num_control_points),
        rng.uniform(0, width, size=num_control_points),
    ]).astype(np.float32)

    num_samples_on_curve = 300
    intensity = float(rng.uniform(120, 200))
    line_half_thickness = int(rng.integers(1, 3))

    for t in np.linspace(0.0, 1.0, num_samples_on_curve):
        point_yx = _evaluate_bezier_at(control_points, t)
        center_y, center_x = int(round(point_yx[0])), int(round(point_yx[1]))
        y_start = max(0, center_y - line_half_thickness)
        y_end = min(height, center_y + line_half_thickness + 1)
        x_start = max(0, center_x - line_half_thickness)
        x_end = min(width, center_x + line_half_thickness + 1)
        if y_start < y_end and x_start < x_end:
            canvas[y_start:y_end, x_start:x_end] = np.maximum(
                canvas[y_start:y_end, x_start:x_end], intensity
            )


# ==================== 数字配置 ====================
def _resize_digit(digit_image, scale):
    """MNIST 数字 (28x28) を指定倍率にリサイズする。

    Args:
        digit_image: (28, 28) uint8 の MNIST 数字
        scale: 拡大倍率 (0.5〜3.0)

    Returns:
        (new_size, new_size) float32 のリサイズ済み数字
    """
    new_size = max(1, int(round(28 * scale)))
    pil_image = Image.fromarray(digit_image)
    resized_pil = pil_image.resize((new_size, new_size), Image.BILINEAR)
    return np.array(resized_pil, dtype=np.float32)


def _place_one_digit(canvas, mask, digit_image, digit_label, top_left_yx, scale):
    """1 個の数字を canvas に描画し、対応する mask チャネルに正解ラベルを書き込む。

    Args:
        canvas: (H, W) float32 の入力画像配列 (in-place 更新)
        mask: (11, H, W) uint8 の multi-hot ラベル配列 (in-place 更新)
        digit_image: (28, 28) uint8 の MNIST 数字
        digit_label: 数字のクラス (0〜9)
        top_left_yx: (y, x) キャンバスでの貼り付け左上座標 (キャンバス外でもよい)
        scale: 拡大倍率
    """
    resized_digit = _resize_digit(digit_image, scale)
    digit_height, digit_width = resized_digit.shape
    canvas_height, canvas_width = canvas.shape

    # キャンバス上の貼り付け範囲を端でクリップする
    y_offset, x_offset = top_left_yx
    canvas_y_start = max(0, y_offset)
    canvas_y_end = min(canvas_height, y_offset + digit_height)
    canvas_x_start = max(0, x_offset)
    canvas_x_end = min(canvas_width, x_offset + digit_width)
    if canvas_y_start >= canvas_y_end or canvas_x_start >= canvas_x_end:
        return  # 完全にキャンバス外なので描画しない

    # 数字側で対応する範囲
    digit_y_start = canvas_y_start - y_offset
    digit_y_end = digit_y_start + (canvas_y_end - canvas_y_start)
    digit_x_start = canvas_x_start - x_offset
    digit_x_end = digit_x_start + (canvas_x_end - canvas_x_start)
    digit_patch = resized_digit[digit_y_start:digit_y_end, digit_x_start:digit_x_end]

    # canvas に max-blend で書き込む (重なっても飽和しない描画)
    canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = np.maximum(
        canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end],
        digit_patch,
    )

    # mask の該当チャネルに 1 を立てる (閾値超え部分のみ)
    digit_pixel_mask = (digit_patch > DIGIT_PIXEL_THRESHOLD).astype(np.uint8)
    mask_channel_index = digit_label + 1  # 0=背景, 1〜10=数字 0〜9
    mask[
        mask_channel_index,
        canvas_y_start:canvas_y_end,
        canvas_x_start:canvas_x_end,
    ] |= digit_pixel_mask  # multi-hot なので OR で重ね合わせる


# ==================== ペア生成 (画像 + マスク) ====================
def generate_canvas_and_mask(
    mnist_images,
    mnist_labels,
    rng,
    digits_per_image_range=DEFAULT_DIGITS_PER_IMAGE_RANGE,
    scale_range=DEFAULT_SCALE_RANGE,
    bezier_count_range=DEFAULT_BEZIER_COUNT_RANGE,
    vertical_streak_count_range=DEFAULT_VERTICAL_STREAK_COUNT_RANGE,
    horizontal_band_count_range=DEFAULT_HORIZONTAL_BAND_COUNT_RANGE,
):
    """1 枚分の (canvas, mask) ペアを生成する。

    画像とマスクの整合性を確実に保つため、両者を 1 つの関数内で同時に作成する。

    Args:
        mnist_images: (N, 28, 28) uint8 の MNIST 数字プール
        mnist_labels: (N,) int の MNIST ラベルプール
        rng: numpy 乱数ジェネレータ (再現性のため呼び出し側でシードを管理する)
        digits_per_image_range: 1 枚に配置する数字個数の (min, max)
        scale_range: 数字スケールの (min, max)
        bezier_count_range: ベジエ曲線本数の (min, max)
        vertical_streak_count_range: 縦ノイズ本数の (min, max)
        horizontal_band_count_range: 横バンドノイズ本数の (min, max)

    Returns:
        canvas: (H, W) uint8 の入力画像
        mask: (11, H, W) uint8 の multi-hot 正解ラベル
    """
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    mask = np.zeros(
        (NUM_CLASSES_WITH_BACKGROUND, CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8
    )

    # ステップ 1: 背景に 1/f ノイズを敷く
    canvas += _generate_one_over_f_noise((CANVAS_HEIGHT, CANVAS_WIDTH), rng)

    # ステップ 2: 紛らわしい背景要素を加える (数字より先に描いて、数字で上書きされるようにする)
    num_streaks = int(rng.integers(
        vertical_streak_count_range[0], vertical_streak_count_range[1] + 1
    ))
    _add_vertical_streaks(canvas, rng, num_streaks)

    num_bands = int(rng.integers(
        horizontal_band_count_range[0], horizontal_band_count_range[1] + 1
    ))
    _add_horizontal_bands(canvas, rng, num_bands)

    num_bezier = int(rng.integers(
        bezier_count_range[0], bezier_count_range[1] + 1
    ))
    for _ in range(num_bezier):
        _draw_bezier_curve(canvas, rng)

    # ステップ 3: 数字を配置する (前景)
    num_digits = int(rng.integers(
        digits_per_image_range[0], digits_per_image_range[1] + 1
    ))
    for _ in range(num_digits):
        digit_index = int(rng.integers(0, len(mnist_images)))
        digit_image = mnist_images[digit_index]
        digit_label = int(mnist_labels[digit_index])
        scale = float(rng.uniform(scale_range[0], scale_range[1]))

        # 配置位置を決める (一部キャンバス外にはみ出すことも許可してデータの多様性を増やす)
        digit_size = max(1, int(round(28 * scale)))
        out_of_canvas_margin = digit_size // 4
        y = int(rng.integers(
            -out_of_canvas_margin,
            CANVAS_HEIGHT - digit_size + out_of_canvas_margin + 1,
        ))
        x = int(rng.integers(
            -out_of_canvas_margin,
            CANVAS_WIDTH - digit_size + out_of_canvas_margin + 1,
        ))
        _place_one_digit(canvas, mask, digit_image, digit_label, (y, x), scale)

    # ステップ 4: canvas を [0, 255] uint8 にクリップする
    canvas_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)

    # ステップ 5: 背景チャネルを決定する (どの数字にも属さないピクセルを背景とする)
    any_digit_present = mask[1:].any(axis=0)
    mask[0] = (~any_digit_present).astype(np.uint8)

    return canvas_uint8, mask


# ==================== データセット一括生成 ====================
def build_dataset(
    output_dir,
    num_images,
    mnist_train=True,
    seed=42,
    verbose=True,
    **generation_kwargs,
):
    """合成データセット (画像 + マスク) を生成してディスクに保存する。

    出力ディレクトリ構造:
        output_dir/
            images/000000.npy   # (H, W) uint8
            images/000001.npy
            ...
            masks/000000.npy    # (11, H, W) uint8
            masks/000001.npy
            ...

    Args:
        output_dir: 出力先ディレクトリ
        num_images: 生成する画像枚数
        mnist_train: True なら MNIST 訓練分 (60000) からサンプリングする。
                     False ならテスト分 (10000) からサンプリングする (汎化評価用)。
        seed: 乱数シード (再現性のため固定)
        verbose: 進捗を print するかどうか
        **generation_kwargs: generate_canvas_and_mask に渡す追加引数
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    mnist_images, mnist_labels = load_mnist_arrays(train=mnist_train)
    rng = np.random.default_rng(seed)

    for index in range(num_images):
        canvas, mask = generate_canvas_and_mask(
            mnist_images, mnist_labels, rng, **generation_kwargs
        )
        np.save(images_dir / f'{index:06d}.npy', canvas)
        np.save(masks_dir / f'{index:06d}.npy', mask)
        if verbose and (index + 1) % 50 == 0:
            print(f'  {index + 1}/{num_images} 枚生成完了')

    if verbose:
        print(f'生成完了: {output_dir} (画像 {num_images} 枚)')


def load_pair(data_dir, index):
    """保存済みの (canvas, mask) ペアを 1 組ロードする。可視化用ユーティリティである。

    Args:
        data_dir: build_dataset で指定した出力先ディレクトリ
        index: 0 始まりのインデックス

    Returns:
        canvas: (H, W) uint8
        mask: (11, H, W) uint8
    """
    data_dir = Path(data_dir)
    canvas = np.load(data_dir / 'images' / f'{index:06d}.npy')
    mask = np.load(data_dir / 'masks' / f'{index:06d}.npy')
    return canvas, mask
