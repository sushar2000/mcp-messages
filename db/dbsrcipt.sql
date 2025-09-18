--CREATE DATABASE WhatsAppDB;


-- USE WhatsAppDB;

-- IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[KKChat]') AND type in (N'U'))
--     DROP TABLE [dbo].[KKChat]
--     CREATE TABLE KKChat (
--         message_datetime DATETIME NOT NULL,
--         message_sender NVARCHAR(100) NOT NULL,
--         message_text NVARCHAR(MAX) NULL,
--         embedding VARBINARY(MAX) NULL
--     );
-- GO

-- Drop the embedding column temporarily
ALTER TABLE KKChat DROP COLUMN embedding;

-- Do the bulk insert
BULK INSERT KKChat
FROM '/home/sushar/KKchat.tsv'
WITH (
    FIELDTERMINATOR = '\t',
    ROWTERMINATOR = '\n',
    FIRSTROW = 0
);

-- Add the embedding column back
ALTER TABLE KKChat ADD embedding VARBINARY(MAX) NULL;




SELECT TOP 10 * FROM KKChat;

-- DELETE FROM KKChat ;

-- sp_help KKChat;


-- SELECT @@VERSION;

-- SELECT OBJECT_ID('COSINE_DISTANCE');


-- 1. Try creating a table with a VECTOR column
-- CREATE TABLE VectorCheck (id INT PRIMARY KEY, emb VECTOR(3));

-- 2. Insert a vector
-- INSERT INTO VectorCheck (id, emb) VALUES (1, VECTOR [0.1, 0.2, 0.3]);

-- 3. Select back
-- SELECT id, emb FROM VectorCheck;

-- 4. Drop it
-- DROP TABLE VectorCheck;

-- EXEC sp_rename 'KKChat.MessageColumn', 'message_text', 'COLUMN';
-- EXEC sp_rename 'KKChat.SenderColumn', 'message_sender', 'COLUMN';
-- EXEC sp_rename 'KKChat.DateColumn', 'message_datetime', 'COLUMN';
EXEC sp_rename 'KKChat', 'KKChatOld', 'OBJECT';

