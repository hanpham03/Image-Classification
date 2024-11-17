import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os


def download_model():
    """
    Tải model phân lớp hình ảnh từ MediaPipe
    """
    model_url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
    model_path = "efficientnet_lite0.tflite"

    if not os.path.exists(model_path):
        print("Đang tải model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Đã tải model thành công!")

    return model_path


def create_classifier(model_path):
    """
    Tạo một Image Classifier từ model đã cho
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageClassifierOptions(base_options=base_options,
                                            max_results=3,
                                            score_threshold=0.3)
    classifier = vision.ImageClassifier.create_from_options(options)
    return classifier


def process_image(image_path, classifier):
    """
    Xử lý và phân lớp một hình ảnh
    """
    # Đọc hình ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc hình ảnh")

    # Chuyển đổi BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tạo đối tượng mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Thực hiện phân lớp
    classification_result = classifier.classify(mp_image)

    return image, classification_result


def draw_results(image, classification_result):
    """
    Vẽ kết quả phân lớp lên hình ảnh
    """
    height, width = image.shape[:2]

    # Tạo một hình ảnh với nền đen để hiển thị kết quả
    result_display = np.zeros((height + 100, width, 3), dtype=np.uint8)
    result_display[0:height, 0:width] = image

    # Vẽ kết quả phân lớp
    y_offset = height + 30
    for category in classification_result.classifications[0].categories:
        label = f"{category.category_name}: {category.score:.2f}"
        cv2.putText(result_display, label,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y_offset += 25

    return result_display


def main():
    # Tải model
    MODEL_PATH = download_model()

    # Nhập đường dẫn hình ảnh
    IMAGE_PATH = input("Nhập đường dẫn đến hình ảnh cần phân lớp: ")

    try:
        # Tạo classifier
        classifier = create_classifier(MODEL_PATH)

        # Xử lý hình ảnh
        image, classification_result = process_image(IMAGE_PATH, classifier)

        # Vẽ kết quả
        result_image = draw_results(image, classification_result)

        # Hiển thị kết quả
        cv2.imshow("Classification Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # In kết quả ra console
        print("\nKết quả phân lớp:")
        for category in classification_result.classifications[0].categories:
            print(f"{category.category_name}: {category.score:.2f}")

    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")


if __name__ == "__main__":
    main()