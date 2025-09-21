IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'WhatsAppDB')
    CREATE DATABASE WhatsAppDB;

USE WhatsAppDB;

-- Create messages table if it doesn't exist
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[messages]') AND type in (N'U'))
    CREATE TABLE messages (
        message_id BIGINT IDENTITY PRIMARY KEY,
        message_datetime DATETIME NOT NULL,
        message_sender NVARCHAR(100) NOT NULL,
        message_text NVARCHAR(MAX) NULL,
        embedding VARBINARY(MAX) NULL
    );

-- Create indexes for optimal query performance

-- 1. Composite Index on (message_datetime, message_sender) - covers most common filter combinations
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_messages_datetime_sender' AND object_id = OBJECT_ID('messages'))
    CREATE NONCLUSTERED INDEX IX_messages_datetime_sender 
    ON messages (message_datetime DESC, message_sender);

-- 2. Index on message_sender - for sender-specific queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_messages_sender' AND object_id = OBJECT_ID('messages'))
    CREATE NONCLUSTERED INDEX IX_messages_sender 
    ON messages (message_sender);

-- 3. Index on message_datetime - for timeline analysis and date filtering
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_messages_datetime' AND object_id = OBJECT_ID('messages'))
    CREATE NONCLUSTERED INDEX IX_messages_datetime 
    ON messages (message_datetime DESC);

-- 4. Covering Index for Embedding Operations - optimizes semantic search
-- Note: Cannot create filtered index on VARBINARY(MAX) column, using regular index instead
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_messages_embedding_coverage' AND object_id = OBJECT_ID('messages'))
    CREATE NONCLUSTERED INDEX IX_messages_embedding_coverage 
    ON messages (message_id)
    INCLUDE (message_text, message_datetime, message_sender, embedding);

-- 5. Full-Text Index for Text Search - COMMENTED OUT: Full-Text Search not installed
-- Note: Full-Text Search is not available on this SQL Server instance
-- You can enable it or use alternative text search strategies
/*
IF NOT EXISTS (SELECT * FROM sys.fulltext_catalogs WHERE name = 'messages_catalog')
    CREATE FULLTEXT CATALOG messages_catalog;

IF NOT EXISTS (SELECT * FROM sys.fulltext_indexes WHERE object_id = OBJECT_ID('messages'))
BEGIN
    DECLARE @pkName NVARCHAR(128)
    SELECT @pkName = name 
    FROM sys.key_constraints 
    WHERE parent_object_id = OBJECT_ID('messages') AND type = 'PK'
    
    DECLARE @sql NVARCHAR(500)
    SET @sql = 'CREATE FULLTEXT INDEX ON messages (message_text) KEY INDEX ' + @pkName + ' ON messages_catalog'
    EXEC sp_executesql @sql
END;
*/

-- 6. Columnstore Index for Analytics - optimizes activity pattern analysis
-- Note: Excluding message_text (NVARCHAR(MAX)) as it cannot participate in columnstore indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_messages_columnstore' AND object_id = OBJECT_ID('messages'))
    CREATE NONCLUSTERED COLUMNSTORE INDEX IX_messages_columnstore
    ON messages (message_datetime, message_sender, message_id);
    
