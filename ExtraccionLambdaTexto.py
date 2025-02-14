import json
import boto3
import os
import pdfplumber
import time
from botocore.config import Config

def lambda_handler(event, context):
    try:
        # Verificar parámetros del evento
        if 'bucket' not in event or 'key' not in event:
            return {'statusCode': 400, 'error': 'Se requieren bucket y key en el evento'}

        bucket_name = event['bucket']
        key = event['key']
        local_pdf = '/tmp/' + os.path.basename(key)

        # Habilitar S3 Transfer Acceleration
        s3 = boto3.client('s3', config=Config(s3={'use_accelerate_endpoint': True}))

        # Descargar archivo desde S3 con aceleración
        print(f"Descargando {key} desde {bucket_name} con Transfer Acceleration...")
        start_time = time.time()
        s3.download_file(bucket_name, key, local_pdf)
        end_time = time.time()
        print(f"Descarga completada en {end_time - start_time:.2f} segundos")

        # Extraer texto del PDF
        extracted_text = []
        with pdfplumber.open(local_pdf) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    extracted_text.append(f"--- Página {page_number} ---\n{text}")

        final_text = "\n\n".join(extracted_text)

        # Eliminar archivo temporal de manera segura
        try:
            os.remove(local_pdf)
        except OSError as e:
            print(f"Error al eliminar {local_pdf}: {str(e)}")

        # Si no hay texto extraído, devolver un error
        if not final_text.strip():
            return {'statusCode': 400, 'error': 'No se pudo extraer texto del PDF'}

        # Conectar a Amazon Bedrock
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")  # O usa "us-west-2"

        # ID CORRECTO del modelo Jamba 1.5 Mini v1
        model_id = "ai21.jamba-1-5-mini-v1:0"

        # Formato de entrada para el modelo (sin límite de tokens)
        payload = {
            "messages": [
                {"role": "user", "content": f"Dame un json estrcuturado bonito solo con todos los  datos financieros , dame un json completo:\n\n{final_text}"}
            ],
            "temperature": 0.7
        }

        # Enviar el texto al modelo de IA
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        # Leer el `StreamingBody` antes de procesarlo
        response_body = json.loads(response["body"].read().decode("utf-8"))

        # Extraer la respuesta generada por el modelo
        generated_text = response_body.get("choices", [{}])[0].get("message", {}).get("content", "")
        generated_text = generated_text.strip()

        # Limpiar delimitadores de bloque si existen
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]
        generated_text = generated_text.strip()

        # Intentar parsear la respuesta a JSON
        try:
            parsed_json = json.loads(generated_text)
        except Exception as e:
            print(f"Error al parsear JSON completo: {str(e)}")
            # Si falla, intentar recortar la cadena hasta la última llave de cierre
            last_brace = generated_text.rfind("}")
            if last_brace != -1:
                candidate = generated_text[:last_brace+1]
                try:
                    parsed_json = json.loads(candidate)
                except Exception as e2:
                    print(f"Error al parsear JSON recortado: {str(e2)}")
                    parsed_json = generated_text  # se deja la cadena original
            else:
                parsed_json = generated_text

        return {
            "statusCode": 200,
            "pdf_text": final_text,
            "ai_response": parsed_json
        }

    except s3.exceptions.ClientError as e:
        print(f"Error al descargar el archivo: {str(e)}")
        return {'statusCode': 500, 'error': f"Error de S3: {str(e)}"}

    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return {'statusCode': 500, 'error': str(e)}

# Solo para pruebas locales
if __name__ == "__main__":
    response = lambda_handler({"bucket": "mi-bucket-financiero", "key": "pruebaa.pdf"}, None)
    print("Texto extraído del PDF:\n", response.get("pdf_text", ""))
    print("\nRespuesta generada por IA:\n", json.dumps(response.get("ai_response", ""), indent=2))
