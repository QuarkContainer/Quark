DROP TABLE FuncAudit;
CREATE TABLE FuncAudit (
    id              UUID NOT NULL PRIMARY KEY,
    jobId           UUID NOT NULL,
    packageName     VARCHAR NOT NULL,
    callerFuncId    UUID,
    funcState       VARCHAR NOT NULL,
    createTime      TIMESTAMP,
    finishTime      TIMESTAMP
    );

//CREATE USER testuser WITH PASSWORD '123456';
//GRANT ALL ON ALL TABLES IN SCHEMA public to testuser;
// https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge