using System.Runtime.CompilerServices;

// Authentication primitives are internal to keep the infrastructure surface small; the test
// projects exercise them directly.
[assembly: InternalsVisibleTo("TokenMiner.UnitTests")]
[assembly: InternalsVisibleTo("TokenMiner.IntegrationTests")]
