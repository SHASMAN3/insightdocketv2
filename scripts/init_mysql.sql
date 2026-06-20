-- InsightDocket MySQL schema
-- Run once to bootstrap the database (SQLAlchemy also creates tables at startup).
-- This file exists for documentation, CI validation, and manual DB setup.

CREATE DATABASE IF NOT EXISTS insightdocket CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE insightdocket;

-- ── documents ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id           INT            NOT NULL AUTO_INCREMENT,
    filename     VARCHAR(512)   NOT NULL,
    s3_key       VARCHAR(1024)  NOT NULL,
    version      INT            NOT NULL DEFAULT 1,
    status       VARCHAR(32)    NOT NULL DEFAULT 'pending'
                     COMMENT 'pending | processing | active | failed | archived',
    page_count   INT            NULL,
    chunk_count  INT            NULL,
    created_at   DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at   DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                     ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_documents_s3_key (s3_key),
    KEY idx_documents_filename (filename),
    KEY idx_documents_status   (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── document_versions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_versions (
    id             INT          NOT NULL AUTO_INCREMENT,
    document_id    INT          NOT NULL,
    version        INT          NOT NULL,
    ingest_status  VARCHAR(32)  NOT NULL DEFAULT 'pending'
                       COMMENT 'pending | processing | success | failed',
    error_message  TEXT         NULL,
    chunk_count    INT          NULL,
    page_count     INT          NULL,
    created_at     DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    KEY idx_docver_document_id (document_id),
    CONSTRAINT fk_docver_document
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── audit_logs ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id                    BIGINT        NOT NULL AUTO_INCREMENT,
    request_id            VARCHAR(64)   NOT NULL,
    document_id           INT           NULL,
    question              TEXT          NOT NULL,
    answer                TEXT          NULL,
    response_type         VARCHAR(32)   NOT NULL DEFAULT 'grounded'
                              COMMENT 'grounded | fallback | injection_blocked | error',
    confidence_score      FLOAT         NULL,
    retrieval_latency_ms  INT           NULL,
    generation_latency_ms INT           NULL,
    total_latency_ms      INT           NULL,
    chunk_ids             JSON          NULL COMMENT 'Array of MongoDB chunk _id strings',
    page_numbers          JSON          NULL COMMENT 'Array of page numbers retrieved',
    injection_detected    TINYINT(1)    NOT NULL DEFAULT 0,
    created_at            DATETIME(6)   NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_audit_request_id (request_id),
    KEY idx_audit_document_id  (document_id),
    KEY idx_audit_created_at   (created_at),
    KEY idx_audit_response_type (response_type),
    CONSTRAINT fk_audit_document
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── api_keys ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
    id               INT          NOT NULL AUTO_INCREMENT,
    name             VARCHAR(128) NOT NULL COMMENT 'Human-readable label for this key',
    key_hash         VARCHAR(64)  NOT NULL COMMENT 'SHA-256 hex digest — raw key never stored',
    rate_limit_rpm   INT          NOT NULL DEFAULT 60,
    is_active        TINYINT(1)   NOT NULL DEFAULT 1,
    created_at       DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_apikeys_key_hash (key_hash),
    KEY idx_apikeys_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
