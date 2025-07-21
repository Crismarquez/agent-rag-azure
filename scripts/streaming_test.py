#!/usr/bin/env python3
"""
test_compliance_stream.py

Instalación previa:
    pip install requests sseclient-py
"""

import json
import requests
from sseclient import SSEClient

def main():
    url = "http://localhost:8000/api/v1/complianceagent"
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    payload = {
        "alert_id": "alert-1234",
        "alert": {
            "alert_id": "alert-1234",
            "type": "transaction",
            "account_id": "ACC-001",
            "monto": 1500,
            "umbral": 1000,
            "generated_at": "2025-07-01T14:00:00Z"
        }
    }

    # Lanza la petición en modo streaming
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()

    # Envuelve la respuesta en SSEClient para iterar eventos
    client = SSEClient(response)

    print("=== Conectado al stream, esperando eventos… ===")
    for event in client.events():
        # Cada event tiene .event (tipo) y .data (payload JSON o texto)
        type_event = event.event
        data = event.data
        try:
            if type_event == "custom":
                data = json.loads(data) # json response from tool
                if data.get("tool") == "assess_risk":
                    output = data.get("output")
                    print(f"[{type_event}] → {output}")
                print(f"[{type_event}] → {data}")
            elif type_event == "messages":
                message = event.data # message from agent string
                print(f"[{type_event}] → {message}")
        except ValueError:
            print("Error al procesar el evento")

if __name__ == "__main__":
    main()
