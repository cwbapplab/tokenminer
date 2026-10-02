using FluentAssertions;
using Microsoft.IdentityModel.JsonWebTokens;
using TokenMiner.Application.Authentication;
using TokenMiner.Domain.Users;
using TokenMiner.Infrastructure.Authentication;
using Xunit;

namespace TokenMiner.UnitTests.Authentication;

public sealed class JwtAccessTokenFactoryTests
{
    private static readonly AuthOptions Options = new()
    {
        Jwt = new AuthOptions.JwtOptions
        {
            SigningKey = "unit-test-signing-key-that-is-at-least-32-chars",
            Issuer = "tokenminer-tests",
            Audience = "tokenminer-tests-client",
            KeyId = "test-key",
            AccessTokenMinutes = 15,
        },
    };

    private static User CreateUser() =>
        new(Guid.NewGuid(), "user@tokenminer.local", "USER@TOKENMINER.LOCAL", "hash", "Test User", DateTimeOffset.UtcNow);

    [Fact]
    public void CreateAccessToken_EmitsSubjectEmailAndRoles()
    {
        var factory = new JwtAccessTokenFactory(Options, TimeProvider.System);
        var user = CreateUser();

        var (token, _) = factory.CreateAccessToken(user, ["user", "admin"]);

        var jwt = new JsonWebTokenHandler().ReadJsonWebToken(token);
        jwt.Subject.Should().Be(user.Id.ToString());
        jwt.GetClaim(AuthClaimTypes.Email).Value.Should().Be(user.Email);
        jwt.GetClaim(AuthClaimTypes.JwtId).Value.Should().NotBeNullOrWhiteSpace();
        jwt.Claims.Where(claim => claim.Type == AuthClaimTypes.Role)
            .Select(claim => claim.Value)
            .Should().BeEquivalentTo(["user", "admin"]);
    }

    [Fact]
    public void CreateAccessToken_SetsIssuerAudienceAndKeyId()
    {
        var factory = new JwtAccessTokenFactory(Options, TimeProvider.System);

        var (token, _) = factory.CreateAccessToken(CreateUser(), ["user"]);

        var jwt = new JsonWebTokenHandler().ReadJsonWebToken(token);
        jwt.Issuer.Should().Be(Options.Jwt.Issuer);
        jwt.Audiences.Should().Contain(Options.Jwt.Audience);
        jwt.Kid.Should().Be(Options.Jwt.KeyId);
    }

    [Fact]
    public void CreateAccessToken_ExpiresAfterConfiguredLifetime()
    {
        var factory = new JwtAccessTokenFactory(Options, TimeProvider.System);

        var (_, expiresInSeconds) = factory.CreateAccessToken(CreateUser(), ["user"]);

        expiresInSeconds.Should().Be(Options.Jwt.AccessTokenMinutes * 60);
    }

    [Fact]
    public void CreateAccessToken_UsesHmacSha256()
    {
        var factory = new JwtAccessTokenFactory(Options, TimeProvider.System);

        var (token, _) = factory.CreateAccessToken(CreateUser(), ["user"]);

        new JsonWebTokenHandler().ReadJsonWebToken(token).Alg.Should().Be("HS256");
    }
}
