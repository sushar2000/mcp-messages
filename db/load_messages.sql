-- Load data from TSV file into the messages table
-- Adjust the file path as necessary for your environment
-- Note: Ensure the SQL Server instance has access to the file path
-- and that the SQL Server service account has read permissions.
-- Using a staging table to facilitate bulk insert
USE WhatsAppDB;

IF OBJECT_ID('tempdb..#messages_staging') IS NOT NULL
    DROP TABLE #messages_staging;

CREATE TABLE #messages_staging (
    message_datetime DATETIME,
    message_sender NVARCHAR(100),
    message_text NVARCHAR(MAX)
);

BULK INSERT #messages_staging
FROM '~/messages1.tsv'
WITH (
    FIELDTERMINATOR = '\t',
    ROWTERMINATOR = '\n',
    FIRSTROW = 1
);

-- INSERT INTO messages (message_datetime, message_sender, message_text)
--     SELECT message_datetime, message_sender, message_text
--     FROM #messages_staging;

DROP TABLE #messages_staging;



