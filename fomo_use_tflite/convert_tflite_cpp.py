def convert_tflite_to_cpp(tflite_path, header_path):
    with open(tflite_path, 'rb') as f:
        bytes_data = f.read()

    with open(header_path, 'w') as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write(f"const unsigned char g_model[] = {{\n")
        
        # Format byte ke bentuk hexadecimal C++
        for i, b in enumerate(bytes_data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
                
        f.write("\n};\n\n")
        f.write(f"const unsigned int g_model_len = {len(bytes_data)};\n\n")
        f.write("#endif // MODEL_DATA_H\n")

    print(f"Berhasil mengonversi model ke: {header_path}")

# Jalankan konversi
convert_tflite_to_cpp("fomo_pure_face_esp32.tflite", "model_data.h")