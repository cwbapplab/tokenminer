using Microsoft.AspNetCore.Identity;
using TokenMiner.Application.Authentication.Abstractions;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>
/// Wraps <see cref="PasswordHasher{TUser}"/> for its PBKDF2 implementation and versioned
/// hash format, without pulling in the Identity user/role stores.
/// </summary>
internal sealed class Pbkdf2PasswordHasher : IPasswordHasher
{
    private static readonly object HashContext = new();

    private readonly PasswordHasher<object> _hasher = new();

    public string Hash(string password) => _hasher.HashPassword(HashContext, password);

    public bool Verify(string password, string passwordHash) =>
        _hasher.VerifyHashedPassword(HashContext, passwordHash, password) != PasswordVerificationResult.Failed;
}
