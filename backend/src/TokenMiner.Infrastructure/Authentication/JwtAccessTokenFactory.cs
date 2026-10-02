using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.JsonWebTokens;
using Microsoft.IdentityModel.Tokens;
using TokenMiner.Application.Authentication;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>Issues short-lived HS256 access tokens carrying subject, email and roles.</summary>
internal sealed class JwtAccessTokenFactory(AuthOptions options, TimeProvider timeProvider) : IAccessTokenFactory
{
    public (string Token, int ExpiresInSeconds) CreateAccessToken(User user, IReadOnlyList<string> roles)
    {
        var now = timeProvider.GetUtcNow();
        var expiresAt = now.AddMinutes(options.Jwt.AccessTokenMinutes);

        var claims = new List<Claim>
        {
            new(AuthClaimTypes.Subject, user.Id.ToString()),
            new(AuthClaimTypes.Email, user.Email),
            new(AuthClaimTypes.JwtId, Guid.NewGuid().ToString()),
        };

        claims.AddRange(roles.Select(role => new Claim(AuthClaimTypes.Role, role)));

        var signingKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(options.Jwt.SigningKey))
        {
            KeyId = options.Jwt.KeyId,
        };

        var descriptor = new SecurityTokenDescriptor
        {
            Issuer = options.Jwt.Issuer,
            Audience = options.Jwt.Audience,
            Subject = new ClaimsIdentity(claims),
            IssuedAt = now.UtcDateTime,
            NotBefore = now.UtcDateTime,
            Expires = expiresAt.UtcDateTime,
            SigningCredentials = new SigningCredentials(signingKey, SecurityAlgorithms.HmacSha256),
        };

        var handler = new JsonWebTokenHandler { SetDefaultTimesOnTokenCreation = false };
        var token = handler.CreateToken(descriptor);

        var expiresInSeconds = (int)Math.Max(0, (expiresAt - now).TotalSeconds);

        return (token, expiresInSeconds);
    }
}
