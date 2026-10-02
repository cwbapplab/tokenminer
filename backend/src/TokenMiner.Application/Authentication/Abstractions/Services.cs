using TokenMiner.Application.Authentication.Models;
using TokenMiner.Domain.Users;

namespace TokenMiner.Application.Authentication.Abstractions;

public interface IPasswordHasher
{
    string Hash(string password);

    bool Verify(string password, string passwordHash);
}

/// <summary>Creates and hashes opaque tokens (refresh tokens, email verification links).</summary>
public interface ISecureTokenService
{
    string GenerateToken();

    string Hash(string token);
}

/// <summary>Generates, hashes and constant-time-verifies numeric OTP codes.</summary>
public interface IOtpService
{
    string GenerateCode();

    string Hash(string code);

    bool Verify(string code, string codeHash);
}

public interface IAccessTokenFactory
{
    /// <returns>The signed JWT and its lifetime in seconds.</returns>
    (string Token, int ExpiresInSeconds) CreateAccessToken(User user, IReadOnlyList<string> roles);
}

public interface IEmailSender
{
    Task SendEmailVerificationAsync(string email, string token, CancellationToken cancellationToken);

    Task SendActivationOtpAsync(string email, string code, CancellationToken cancellationToken);
}

public interface IGoogleTokenValidator
{
    /// <returns>Null when the token is invalid or not intended for our client id.</returns>
    Task<GoogleUserInfo?> ValidateAsync(string idToken, CancellationToken cancellationToken);
}

/// <summary>Issues a fresh access + refresh pair and stages the refresh token for persistence.</summary>
public interface IAuthTokenIssuer
{
    Task<IssuedTokenPair> IssueAsync(
        User user,
        IReadOnlyList<string> roles,
        string? ip,
        string? userAgent,
        CancellationToken cancellationToken);
}
