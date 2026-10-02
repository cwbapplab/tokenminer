using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Domain.Users;

namespace TokenMiner.Application.Authentication.Services;

/// <summary>
/// Builds an access + refresh pair. The refresh token is generated here but only
/// persisted when the caller saves the unit of work, so a handler can commit the
/// user and token changes together.
/// </summary>
public sealed class AuthTokenIssuer(
    IAccessTokenFactory accessTokenFactory,
    IRefreshTokenStore refreshTokenStore,
    ISecureTokenService secureTokenService,
    TimeProvider timeProvider,
    AuthOptions options) : IAuthTokenIssuer
{
    public Task<IssuedTokenPair> IssueAsync(
        User user,
        IReadOnlyList<string> roles,
        string? ip,
        string? userAgent,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var (accessToken, expiresInSeconds) = accessTokenFactory.CreateAccessToken(user, roles);

        var rawRefreshToken = secureTokenService.GenerateToken();
        var now = timeProvider.GetUtcNow();

        var refreshToken = new RefreshToken(
            Guid.NewGuid(),
            user.Id,
            secureTokenService.Hash(rawRefreshToken),
            now.AddDays(options.RefreshTokenLifetimeDays),
            now,
            ip,
            userAgent);

        refreshTokenStore.Add(refreshToken);

        var tokens = new AuthTokens(accessToken, rawRefreshToken, expiresInSeconds);
        return Task.FromResult(new IssuedTokenPair(tokens, refreshToken.Id));
    }
}
