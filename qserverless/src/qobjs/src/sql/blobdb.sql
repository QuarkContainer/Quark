DROP DATABASE blobdb;
CREATE DATABASE blobdb;

DROP TABLE blobtbl;
CREATE TABLE blobtbl (
    id              UUID NOT NULL PRIMARY KEY,
    data            bytea,
    createTime      TIMESTAMP
);

CREATE USER blob_user WITH PASSWORD '123456';
GRANT ALL ON ALL TABLES IN SCHEMA public to blob_user;