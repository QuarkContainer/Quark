DROP DATABASE auditdb;
CREATE DATABASE auditdb;

DROP TABLE FuncAudit;
CREATE TABLE FuncAudit (
    id              UUID NOT NULL PRIMARY KEY,
    jobId           UUID NOT NULL,
    namespace       VARCHAR NOT NULL,
    packageName     VARCHAR NOT NULL,
    funcName        VARCHAR NOT NULL,
    callerFuncId    VARCHAR NOT NULL,
    funcState       VARCHAR NOT NULL,
    nodeId          VARCHAR,
    createTime      TIMESTAMP,
    assignedTime    TIMESTAMP,
    finishTime      TIMESTAMP
);

CREATE USER audit_user WITH PASSWORD '123456';
GRANT ALL ON ALL TABLES IN SCHEMA public to audit_user;

// https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge