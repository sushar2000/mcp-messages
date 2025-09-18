--CREATE DATABASE WhatsAppDB;


USE WhatsAppDB;

-- IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[KKChat]') AND type in (N'U'))
--     DROP TABLE [dbo].[KKChat]
--     CREATE TABLE KKChat (
--         DateColumn DATETIME,
--         SenderColumn NVARCHAR(100),
--         MessageColumn NVARCHAR(MAX)
--     );
-- GO

-- ALTER TABLE KKChat
-- ADD embedding VARBINARY(MAX);

-- BULK INSERT KKChat
-- FROM '/var/opt/mssql/data/KKchat.tsv'
-- WITH (
--     FIELDTERMINATOR = '\t',  -- Tab-separated
--     ROWTERMINATOR = '\n',
--     FIRSTROW = 0             -- Skip header row if present
-- );


SELECT TOP 10 * FROM KKChat;

sp_help KKChat;


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

