# memory-lancedb

OpenClaw LanceDB-backed long-term memory plugin with ARK embedding support.

## Features

- LanceDB vector storage for semantic memory
- Support for OpenAI and Volcengine ARK embedding providers
- Auto-capture and auto-recall of conversation memories
- Multimodal embedding support (text + image)

## Configuration

```json
{
  "plugins": {
    "memory-lancedb": {
      "enabled": true,
      "embedding": {
        "provider": "volcengine",
        "model": "doubao-embedding-vision-251215"
      }
    }
  }
}
```

## Supported Providers

- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`
- **Volcengine ARK**: `doubao-embedding-vision-251215`, `doubao-embedding-text-240515`, etc.
