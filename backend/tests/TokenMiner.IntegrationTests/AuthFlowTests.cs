using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using TokenMiner.Contracts.Auth;
using Xunit;

namespace TokenMiner.IntegrationTests;

[Collection(ApiCollection.Name)]
public sealed class AuthFlowTests(ApiFactory factory)
{
    private const string ValidPassword = "Str0ngPassword!23";

    private static string NewEmail() => $"user-{Guid.NewGuid():N}@tokenminer.local";

    [Fact]
    public async Task FullFlow_Activation_RefreshRotation_AndReuseDetection()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        var register = await client.PostAsJsonAsync(
            "/api/auth/register",
            new RegisterRequest(email, ValidPassword, "Flow User"));

        register.StatusCode.Should().Be(HttpStatusCode.Accepted);
        var pending = await register.Content.ReadFromJsonAsync<PendingActivationResponse>();
        pending!.RequiresActivation.Should().BeTrue();

        var verifyEmail = await client.PostAsJsonAsync(
            "/api/auth/verify-email",
            new VerifyEmailRequest(factory.Email.LatestVerificationToken(email)));
        verifyEmail.StatusCode.Should().Be(HttpStatusCode.OK);

        var activationCode = factory.Email.LatestActivationCode(email);
        var wrongCode = activationCode == "000000" ? "000001" : "000000";

        var wrongOtp = await client.PostAsJsonAsync(
            "/api/auth/otp/verify",
            new VerifyOtpRequest(email, wrongCode));
        wrongOtp.StatusCode.Should().Be(HttpStatusCode.Unauthorized);

        var activate = await client.PostAsJsonAsync(
            "/api/auth/otp/verify",
            new VerifyOtpRequest(email, activationCode));
        activate.StatusCode.Should().Be(HttpStatusCode.OK);

        var tokens = await activate.Content.ReadFromJsonAsync<AuthTokensResponse>();
        tokens.Should().NotBeNull();
        tokens!.AccessToken.Should().NotBeNullOrWhiteSpace();
        tokens.RefreshToken.Should().NotBeNullOrWhiteSpace();
        tokens.ExpiresIn.Should().Be(15 * 60);

        using var meRequest = new HttpRequestMessage(HttpMethod.Get, "/api/auth/me");
        meRequest.Headers.Authorization = new AuthenticationHeaderValue("Bearer", tokens.AccessToken);
        var me = await client.SendAsync(meRequest);

        me.StatusCode.Should().Be(HttpStatusCode.OK);
        var profile = await me.Content.ReadFromJsonAsync<UserProfileResponse>();
        profile!.Email.Should().Be(email);
        profile.Status.Should().Be("active");
        profile.EmailConfirmed.Should().BeTrue();
        profile.Roles.Should().Contain("user");

        var refresh = await client.PostAsJsonAsync("/api/auth/refresh", new RefreshTokenRequest(tokens.RefreshToken));
        refresh.StatusCode.Should().Be(HttpStatusCode.OK);
        var rotated = await refresh.Content.ReadFromJsonAsync<AuthTokensResponse>();
        rotated!.RefreshToken.Should().NotBe(tokens.RefreshToken);

        // A second, independent sign-in so the reuse below has a bystander session to spare.
        var second = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, ValidPassword));
        second.StatusCode.Should().Be(HttpStatusCode.OK);
        var secondTokens = await second.Content.ReadFromJsonAsync<AuthTokensResponse>();

        // Replaying the superseded token is treated as theft.
        var reuse = await client.PostAsJsonAsync("/api/auth/refresh", new RefreshTokenRequest(tokens.RefreshToken));
        reuse.StatusCode.Should().Be(HttpStatusCode.Unauthorized);

        // The replay closes the session it came from: the token the replayed one rotated into.
        var afterReuse = await client.PostAsJsonAsync("/api/auth/refresh", new RefreshTokenRequest(rotated.RefreshToken));
        afterReuse.StatusCode.Should().Be(HttpStatusCode.Unauthorized);

        // ...but stops there. The other session is untouched, which is the property that keeps a
        // shared account from logging every device out when one client replays an old token.
        var bystander = await client.PostAsJsonAsync("/api/auth/refresh", new RefreshTokenRequest(secondTokens!.RefreshToken));
        bystander.StatusCode.Should().Be(HttpStatusCode.OK);

        var login = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, ValidPassword));
        login.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Fact]
    public async Task ConcurrentRefreshes_OfTheSameToken_OnlyOneSucceeds()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        await client.PostAsJsonAsync("/api/auth/register", new RegisterRequest(email, ValidPassword, null));
        await client.PostAsJsonAsync("/api/auth/verify-email", new VerifyEmailRequest(factory.Email.LatestVerificationToken(email)));
        var activate = await client.PostAsJsonAsync(
            "/api/auth/otp/verify",
            new VerifyOtpRequest(email, factory.Email.LatestActivationCode(email)));
        var tokens = await activate.Content.ReadFromJsonAsync<AuthTokensResponse>();

        // Two clients hammer the same refresh token at once. Without an atomic rotation both would
        // mint a pair and both would stay active; exactly one may win.
        var first = factory.CreateClient();
        var second = factory.CreateClient();

        var responses = await Task.WhenAll(
            first.PostAsJsonAsync("/api/auth/refresh", new RefreshTokenRequest(tokens!.RefreshToken)),
            second.PostAsJsonAsync("/api/auth/refresh", new RefreshTokenRequest(tokens.RefreshToken)));

        responses.Count(response => response.StatusCode == HttpStatusCode.OK).Should().Be(1);
        responses.Count(response => response.StatusCode == HttpStatusCode.Unauthorized).Should().Be(1);
    }

    [Fact]
    public async Task Register_WithExistingEmail_ReturnsConflict()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        var first = await client.PostAsJsonAsync("/api/auth/register", new RegisterRequest(email, ValidPassword, null));
        first.StatusCode.Should().Be(HttpStatusCode.Accepted);

        var second = await client.PostAsJsonAsync("/api/auth/register", new RegisterRequest(email, ValidPassword, null));

        second.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }

    [Fact]
    public async Task Register_WithWeakPassword_ReturnsValidationProblem()
    {
        var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync(
            "/api/auth/register",
            new RegisterRequest(NewEmail(), "short", null));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);

        var problem = await response.Content.ReadFromJsonAsync<JsonElement>();
        problem.TryGetProperty("errors", out var errors).Should().BeTrue();
        errors.TryGetProperty("Password", out var passwordErrors).Should().BeTrue();
        passwordErrors.GetArrayLength().Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task Login_BeforeActivation_ReturnsForbidden()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        await client.PostAsJsonAsync("/api/auth/register", new RegisterRequest(email, ValidPassword, null));

        var login = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, ValidPassword));

        login.StatusCode.Should().Be(HttpStatusCode.Forbidden);
    }

    [Fact]
    public async Task Login_WithWrongPassword_ReturnsUnauthorized()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        await client.PostAsJsonAsync("/api/auth/register", new RegisterRequest(email, ValidPassword, null));
        await client.PostAsJsonAsync("/api/auth/otp/verify", new VerifyOtpRequest(email, factory.Email.LatestActivationCode(email)));

        var login = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, "Wr0ngPassword!23"));

        login.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task Login_LocksOutAfterMaxFailedAttempts()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        // Configured maximum is 3 failed attempts within the window.
        for (var attempt = 0; attempt < 3; attempt++)
        {
            var failed = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, "Wr0ngPassword!23"));
            failed.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
        }

        var locked = await client.PostAsJsonAsync("/api/auth/login", new LoginRequest(email, "Wr0ngPassword!23"));

        locked.StatusCode.Should().Be(HttpStatusCode.TooManyRequests);
    }

    [Fact]
    public async Task Me_WithoutToken_ReturnsUnauthorized()
    {
        var client = factory.CreateClient();

        var response = await client.GetAsync("/api/auth/me");

        response.StatusCode.Should().Be(HttpStatusCode.Unauthorized);
    }

    [Fact]
    public async Task RequestOtp_WithinCooldown_ReturnsTooManyRequests()
    {
        var client = factory.CreateClient();
        var email = NewEmail();

        await client.PostAsJsonAsync("/api/auth/register", new RegisterRequest(email, ValidPassword, null));

        var resend = await client.PostAsJsonAsync("/api/auth/otp/request", new RequestOtpRequest(email));

        resend.StatusCode.Should().Be(HttpStatusCode.TooManyRequests);
    }

    [Fact]
    public async Task MalformedJsonBody_ReturnsBadRequest()
    {
        var client = factory.CreateClient();

        using var content = new StringContent("{\"email\":123,\"password\":true}", Encoding.UTF8, "application/json");
        var response = await client.PostAsync("/api/auth/login", content);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }
}
