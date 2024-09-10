import cv2
import grpc
from concurrent import futures

import numpy as np

import image_processor_pb2
import image_processor_pb2_grpc
from PIL import Image
import io

class ImageProcessorServicer(image_processor_pb2_grpc.ImageProcessorServicer):
    def ProcessImage(self, request, context):
        # 클라이언트로부터 이미지를 받음
        image_data = request.image

        # 이미지 처리(예: 리사이즈)
        processed_image_data = self.preprocess_image(image_data)

        # 처리된 이미지를 반환
        return image_processor_pb2.ImageResponse(processed_image=processed_image_data)

    def preprocess_image(self, image_data):
        np_arr = np.frombuffer(image_data, dtype=np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        adaptive_binary_image = cv2.adaptiveThreshold(
            gray_image,  # 입력 이미지 (그레이스케일)
            255,  # 임계값을 초과하는 픽셀에 할당할 값 (흰색)
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 임계값 계산 방법 (가우시안 가중치)
            cv2.THRESH_BINARY,  # 이진화 방식
            21,  # 블록 크기 (이웃할 픽셀들의 크기, 홀수만 가능)
            10,  # 평균에서 뺄 값 (값이 클수록 더 어두운 부분도 흰색으로 변환됨)
        )
        binary_pil_img = Image.fromarray(adaptive_binary_image).convert("1")
        # 이미지를 바이트로 다시 변환
        byte_arr = io.BytesIO()
        # Group 4 압축 적용하며 TIFF 형식으로 저장
        binary_pil_img.save(byte_arr, format='TIFF', compression='group4')
        # 바이트 데이터를 반환
        return byte_arr.getvalue()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    image_processor_pb2_grpc.add_ImageProcessorServicer_to_server(ImageProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC 서버가 실행 중입니다...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
