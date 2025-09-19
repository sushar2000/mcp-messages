
USE WhatsAppDB;




-- -- Drop table if exists and create new one with updated structure
-- IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[KKChat]') AND type in (N'U'))
--     DROP TABLE [dbo].[KKChat]

-- CREATE TABLE KKChat (
--     message_id BIGINT IDENTITY PRIMARY KEY,
--     message_datetime DATETIME NOT NULL,
--     message_sender NVARCHAR(100) NOT NULL,
--     message_text NVARCHAR(MAX) NULL,
--     embedding VARBINARY(MAX) NULL
-- );
-- GO

-- -- Load data from TSV file into the KKChat table
-- -- Adjust the file path as necessary for your environment
-- -- Note: Ensure the SQL Server instance has access to the file path
-- -- and that the SQL Server service account has read permissions.
-- -- Using a staging table to facilitate bulk insert
-- IF OBJECT_ID('tempdb..#KKChat_Staging') IS NOT NULL
--     DROP TABLE #KKChat_Staging;
-- CREATE TABLE #KKChat_Staging (
--     message_datetime DATETIME,
--     message_sender NVARCHAR(100),
--     message_text NVARCHAR(MAX)
-- );

-- BULK INSERT #KKChat_Staging
-- FROM '/home/sushar/KKchat.tsv'
-- WITH (
--     FIELDTERMINATOR = '\t',
--     ROWTERMINATOR = '\n',
--     FIRSTROW = 1
-- );

-- INSERT INTO KKChat (message_datetime, message_sender, message_text)
--     SELECT message_datetime, message_sender, message_text
--     FROM #KKChat_Staging;

-- DROP TABLE #KKChat_Staging;





SELECT TOP 10 * FROM KKChat ORDER BY message_datetime asc;

-- DELETE FROM KKChat ;

-- sp_help KKChat;




-- SELECT @@VERSION;

-- SELECT OBJECT_ID('COSINE_DISTANCE');


-- -- 1. Try creating a table with a VECTOR column
-- if not exists (select * from sysobjects where name='VectorCheck' and xtype='U')
--     CREATE TABLE VectorCheck (id INT PRIMARY KEY, emb VECTOR(3));

-- -- 2. Insert a vector
-- INSERT INTO VectorCheck (id, emb) VALUES 
--   (1, '[0.1, 2, 30]'),
--   (2, '[-100.2, 0.123, 9.876]'),
--   (3, JSON_ARRAY(1.0, 2.0, 3.0)); -- Using JSON_ARRAY to create a vector

-- -- 3. Select back
-- SELECT id, emb FROM VectorCheck;

-- -- 4. Drop it
-- DROP TABLE VectorCheck;

-- Query to check for messages without embeddings
SELECT COUNT(*) 
FROM KKChat 
WHERE embedding IS NULL 
