namespace TokenMiner.Application.Authentication;

/// <summary>
/// Bound from the <c>Auth</c> configuration section. Kept as a plain POCO so the
/// Application layer needs no dependency on the options/configuration packages.
/// </summary>
public sealed class AuthOptions
{
    public const string SectionName = "Auth";

    public JwtOptions Jwt { get; set; } = new();

    public OtpOptions Otp { get; set; } = new();

    public GoogleOptions Google { get; set; } = new();

    public LockoutOptions Lockout { get; set; } = new();

    public RateLimitOptions RateLimit { get; set; } = new();

    /// <summary>Lifetime of an issued refresh token.</summary>
    public int RefreshTokenLifetimeDays { get; set; } = 30;

    /// <summary>
    /// Template for the email validation link. <c>{token}</c> is replaced with the raw token.
    /// Defaults to a custom URI scheme suitable for a desktop client.
    /// </summary>
    public string EmailVerificationUrlTemplate { get; set; } = "tokenminer://verify-email?token={token}";

    /// <summary>When true, Google sign-in still requires OTP activation before tokens are issued.</summary>
    public bool RequireOtpForOauth { get; set; } = true;

    /// <summary>
    /// Accounts granted the admin role at startup. Empty by default. The account must already
    /// exist, so the flow is: register, set this, restart.
    /// </summary>
    public List<string> BootstrapAdminEmails { get; set; } = [];

    public sealed class JwtOptions
    {
        public string SigningKey { get; set; } = string.Empty;

        public string Issuer { get; set; } = "tokenminer";

        public string Audience { get; set; } = "tokenminer-client";

        /// <summary>Key identifier emitted in the JWT header, to allow future key rotation.</summary>
        public string KeyId { get; set; } = "default";

        public int AccessTokenMinutes { get; set; } = 15;
    }

    public sealed class OtpOptions
    {
        public int Length { get; set; } = 6;

        public int ExpiryMinutes { get; set; } = 10;

        public int MaxAttempts { get; set; } = 5;

        /// <summary>Minimum delay between OTP resends for the same user and purpose.</summary>
        public int ResendCooldownSeconds { get; set; } = 60;

        /// <summary>Key used to HMAC codes before storage. Must be set outside development.</summary>
        public string HmacKey { get; set; } = string.Empty;

        /// <summary>Lifetime of the emailed verification link.</summary>
        public int EmailVerificationTokenLifetimeHours { get; set; } = 24;
    }

    public sealed class GoogleOptions
    {
        /// <summary>Google OAuth client id of the desktop app; validated as the token audience.</summary>
        public string DesktopClientId { get; set; } = string.Empty;
    }

    public sealed class LockoutOptions
    {
        public int MaxFailedAttempts { get; set; } = 5;

        public int WindowMinutes { get; set; } = 15;
    }

    /// <summary>Per-IP throttle applied to the anonymous authentication endpoints.</summary>
    public sealed class RateLimitOptions
    {
        public int PermitLimit { get; set; } = 30;

        public int WindowSeconds { get; set; } = 60;
    }
}
