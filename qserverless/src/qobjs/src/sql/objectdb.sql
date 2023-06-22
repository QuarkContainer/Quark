DROP DATABASE blobdb;
CREATE DATABASE blobdb;

DROP TABLE objecttbl;
CREATE TABLE objecttbl (
    namespace       VARCHAR NOT NULL,
    name            VARCHAR NOT NULL,
    data            bytea,
    createTime      TIMESTAMP
);

ALTER TABLE objecttbl ADD CONSTRAINT "ID_PKEY" PRIMARY KEY (namespace, name);

CREATE USER blob_user WITH PASSWORD '123456';
GRANT ALL ON ALL TABLES IN SCHEMA public to blob_user;

//ALTER TABLE objecttbl OWNER TO postgres;
