import requests
import json
import time

# Set url
FLASK_APP_URL = "http://127.0.0.1:8000/predict"

# Data input untuk inferensi
INPUT_DATA_SAMPLES = [
    {
        "dataframe_split": {
            "columns": [
                "MonthlyIncome",
                "Age",
                "TotalWorkingYears",
                "OverTime",
                "MonthlyRate",
                "DailyRate",
                "EmployeeId",
                "DistanceFromHome",
                "HourlyRate",
                "NumCompaniesWorked"
            ],
            "data": [
                [
                    0.0783043707214323,
                    0.3571428571428571,
                    0.175,
                    0,
                    0.1837382051796827,
                    0.5884037222619899,
                    1140,
                    0.0357142857142857,
                    0.6857142857142857,
                    0.4444444444444444
                ]
            ]
        }
    },
    {
        "dataframe_split": {
            "columns": [
                "MonthlyIncome",
                "Age",
                "TotalWorkingYears",
                "OverTime",
                "MonthlyRate",
                "DailyRate",
                "EmployeeId",
                "DistanceFromHome",
                "HourlyRate",
                "NumCompaniesWorked"
            ],
            "data": [
                [
                    0.05,
                    0.28,
                    0.10,
                    0,
                    0.15,
                    0.40,
                    2001,
                    0.02,
                    0.55,
                    0.22
                ]
            ]
        }
    },
    # data bisa ditambahkan lagi
]

# Jumlah total request yang dikirim dalam rentang 0.05
NUM_REQUESTS_TO_SEND = 10
REQUEST_DELAY_SECONDS = 0.05

def send_inference_request(data_payload):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(FLASK_APP_URL, headers=headers, data=json.dumps(data_payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print(f"Error koneksi ke Flask: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP dari Flask: {e.response.status_code} - {e.response.text}")
        return None
    except json.JSONDecodeError:
        print(f"Gagal mengurai respons JSON: {response.text}")
        return None
    except Exception as e:
        print(f"Terjadi error tak terduga: {e}")
        return None

if __name__ == "__main__":
    print(f"Mengirim {NUM_REQUESTS_TO_SEND} permintaan ke {FLASK_APP_URL}...")
    successful_requests = 0
    
    # Looping untuk mengirim sejumlah request yang diinginkan
    for i in range(NUM_REQUESTS_TO_SEND):
        # Menggunakan sampel data secara bergantian
        data_index = i % len(INPUT_DATA_SAMPLES)
        current_data = INPUT_DATA_SAMPLES[data_index]
        
        print(f"[{i+1}/{NUM_REQUESTS_TO_SEND}] Mengirim request dengan sampel data {data_index + 1}...")
        
        response_data = send_inference_request(current_data)
        
        if response_data:
            successful_requests += 1
            # print(f"Respons sukses: {response_data}")
        else:
            print(f"Request {i+1} gagal.")
            
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nSelesai mengirim {NUM_REQUESTS_TO_SEND} permintaan.")
    print(f"Jumlah permintaan sukses: {successful_requests}")
