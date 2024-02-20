
import asyncio

import qserverless

if __name__ == '__main__':
    namespace = qserverless.client.GetNamespaceFromEnvVar()
    packageId = qserverless.client.GetPackageIdFromEnvVar()
    svcAddr = qserverless.client.GetNodeAgentAddrFromEnvVar()
    print("namespace = ", namespace, " packageId =", packageId, " svcAddr = ", svcAddr)
    qserverless.client.Register(svcAddr, namespace, packageId, False)
    asyncio.run(qserverless.client.StartSvc())