---
title: Inventel AI Service
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Inventel AI Service

FastAPI service untuk FunctionGemma bot inventory gudang.

## Endpoint

### POST `/predict`

Terima pesan user, return function call yang sesuai.

**Request:**

```json
{
  "message": "laptop ada berapa yang tersedia?",
  "userId": "U001"
}
```

**Response (function call):**

```json
{
  "type": "function_call",
  "calls": [
    {
      "name": "getItemStock",
      "arguments": { "keyword": "laptop" }
    }
  ]
}
```

**Response (teks biasa):**

```json
{
  "type": "text",
  "text": "Halo! Ada yang bisa saya bantu?"
}
```

### GET `/health`

Health check endpoint.
