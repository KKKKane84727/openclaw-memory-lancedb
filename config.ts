import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type EmbeddingProvider = "openai" | "volcengine";

export type MemoryConfig = {
  embedding: {
    provider: EmbeddingProvider;
    model: string;
    apiKey: string;
    baseUrl?: string;
    dimensions?: number;
  };
  dbPath?: string;
  autoCapture?: boolean;
  autoRecall?: boolean;
  captureMaxChars?: number;
};

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

const DEFAULT_MODEL = "text-embedding-3-small";
export const DEFAULT_CAPTURE_MAX_CHARS = 500;
const LEGACY_STATE_DIRS: string[] = [];

function resolveDefaultDbPath(): string {
  const home = homedir();
  const preferred = join(home, ".openclaw", "memory", "lancedb");
  try {
    if (fs.existsSync(preferred)) {
      return preferred;
    }
  } catch {
    // best-effort
  }

  for (const legacy of LEGACY_STATE_DIRS) {
    const candidate = join(home, legacy, "memory", "lancedb");
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // best-effort
    }
  }

  return preferred;
}

const DEFAULT_DB_PATH = resolveDefaultDbPath();

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  // OpenAI models
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  // Volcengine ARK models (text)
  "doubao-embedding-text-240515": 1024,
  "doubao-embedding-text-240715": 1024,
  "doubao-embedding-large-text-240915": 2048,
  "doubao-embedding-large-text-250515": 2048,
  // Volcengine ARK models (vision/multimodal)
  "doubao-embedding-vision-241215": 1280,
  "doubao-embedding-vision-250328": 1280,
  "doubao-embedding-vision-250615": 1280,
  "doubao-embedding-vision-251215": 2048,
};

// Default ARK base URL
const DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3";

// Check if running in Cartooner environment
function isCartoonerEnvironment(): boolean {
  return process.env.CARTOONER_HOME !== undefined || process.env.ARK_API_KEY !== undefined;
}

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

export function vectorDimsForModel(model: string): number {
  // Extract base model name (handle provider prefixes like "doubao-embedding-")
  const baseModel = model.includes("-") ? model : model;
  const dims = EMBEDDING_DIMENSIONS[baseModel];
  if (!dims) {
    throw new Error(
      `Unsupported embedding model: ${model}. Supported: ${Object.keys(EMBEDDING_DIMENSIONS).join(", ")}`,
    );
  }
  return dims;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function resolveEmbeddingModel(embedding: Record<string, unknown>): string {
  const model = typeof embedding.model === "string" ? embedding.model : DEFAULT_MODEL;
  if (typeof embedding.dimensions !== "number") {
    vectorDimsForModel(model);
  }
  return model;
}

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["embedding", "dbPath", "autoCapture", "autoRecall", "captureMaxChars"],
      "memory config",
    );

    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding || typeof embedding.apiKey !== "string") {
      throw new Error("embedding.apiKey is required");
    }
    assertAllowedKeys(
      embedding,
      ["provider", "apiKey", "model", "baseUrl", "dimensions"],
      "embedding config",
    );

    // Determine provider (default: openai)
    const provider = (embedding.provider as EmbeddingProvider) || "openai";
    if (provider !== "openai" && provider !== "volcengine") {
      throw new Error(`Unsupported embedding provider: ${provider}. Supported: openai, volcengine`);
    }

    // For volcengine provider, auto-fill from environment if not explicitly provided
    let resolvedApiKey = embedding.apiKey;
    let resolvedBaseUrl = embedding.baseUrl;

    if (provider === "volcengine") {
      // Check for ARK_API_KEY in environment (Cartooner compatibility)
      const envApiKey = process.env.ARK_API_KEY;
      if (envApiKey && !String(embedding.apiKey).startsWith("${")) {
        // Explicit key takes precedence
        resolvedApiKey = embedding.apiKey;
      } else if (envApiKey) {
        resolvedApiKey = envApiKey;
      }

      // Check for ARK_BASE_URL in environment
      const envBaseUrl = process.env.ARK_BASE_URL;
      if (envBaseUrl && !embedding.baseUrl) {
        resolvedBaseUrl = envBaseUrl;
      } else if (embedding.baseUrl) {
        resolvedBaseUrl = embedding.baseUrl;
      }
    }

    const model = resolveEmbeddingModel(embedding);

    const captureMaxChars =
      typeof cfg.captureMaxChars === "number" ? Math.floor(cfg.captureMaxChars) : undefined;
    if (
      typeof captureMaxChars === "number" &&
      (captureMaxChars < 100 || captureMaxChars > 10_000)
    ) {
      throw new Error("captureMaxChars must be between 100 and 10000");
    }

    return {
      embedding: {
        provider,
        model,
        apiKey: resolveEnvVars(resolvedApiKey),
        baseUrl:
          typeof resolvedBaseUrl === "string"
            ? resolveEnvVars(resolvedBaseUrl)
            : provider === "volcengine"
              ? DEFAULT_ARK_BASE_URL
              : undefined,
        dimensions: typeof embedding.dimensions === "number" ? embedding.dimensions : undefined,
      },
      dbPath: typeof cfg.dbPath === "string" ? cfg.dbPath : DEFAULT_DB_PATH,
      autoCapture: cfg.autoCapture === true,
      autoRecall: cfg.autoRecall !== false,
      captureMaxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
    };
  },
  uiHints: {
    "embedding.provider": {
      label: "Embedding Provider",
      help: "Choose the embedding provider",
    },
    "embedding.apiKey": {
      label: "API Key",
      sensitive: true,
      placeholder: "sk-proj-... / ARK_API_KEY",
      help: "API key for the embedding provider (use ${YOUR_API_KEY} env var)",
    },
    "embedding.baseUrl": {
      label: "Base URL",
      placeholder: "https://api.openai.com/v1",
      help: "Base URL for compatible providers (e.g. http://localhost:11434/v1). For ARK use: https://ark.cn-beijing.volces.com/api/v3",
      advanced: true,
    },
    "embedding.dimensions": {
      label: "Dimensions",
      placeholder: "1536",
      help: "Vector dimensions for custom models (required for non-standard models)",
      advanced: true,
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_MODEL,
      help: "Embedding model to use (e.g. text-embedding-3-small, doubao-embedding-text-240515)",
    },
    dbPath: {
      label: "Database Path",
      placeholder: "~/.openclaw/memory/lancedb",
      advanced: true,
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information from conversations",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
    captureMaxChars: {
      label: "Capture Max Chars",
      help: "Maximum message length eligible for auto-capture",
      advanced: true,
      placeholder: String(DEFAULT_CAPTURE_MAX_CHARS),
    },
  },
};
