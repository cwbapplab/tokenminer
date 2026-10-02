namespace TokenMiner.Contracts.Auth;

/// <summary>Access + refresh token pair returned by every successful sign-in/presentation of credentials.</summary>
public sealed record AuthTokensResponse(string AccessToken, string RefreshToken, int ExpiresIn);

/// <summary>Returned when the account exists but still needs OTP activation before tokens are issued.</summary>
public sealed record PendingActivationResponse(bool RequiresActivation);

public sealed record UserProfileResponse(
    Guid Id,
    string Email,
    string? DisplayName,
    string Status,
    bool EmailConfirmed,
    IReadOnlyList<string> Roles,
    DateTimeOffset CreatedAt);
