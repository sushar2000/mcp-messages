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
