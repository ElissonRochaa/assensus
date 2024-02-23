import cv2

# Caminho para o vídeo de entrada e saída
video_input_path = "people.mp4"
video_output_path = "people_384_640.mp4"

# Dimensões desejadas para o vídeo redimensionado
new_width = 640
new_height = 384

# Abra o vídeo de entrada
cap = cv2.VideoCapture(video_input_path)

# Verifique se o vídeo de entrada foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo de entrada")
    exit()

# Obtenha as informações do vídeo de entrada
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Crie um objeto VideoWriter para escrever o vídeo redimensionado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (new_width, new_height))

# Processamento do vídeo frame a frame
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Redimensionar o quadro para a nova largura e altura
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Escrever o quadro redimensionado no vídeo de saída
        out.write(resized_frame)

        # Mostrar o progresso
        cv2.imshow('Resized Video', resized_frame)

        # Verifique se o usuário pressionou a tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Libere os recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print("Redimensionamento do vídeo concluído.")