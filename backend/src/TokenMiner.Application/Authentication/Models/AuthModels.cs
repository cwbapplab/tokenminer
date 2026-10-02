namespace TokenMiner.Application.Authentication.Models;

/// <summary>Token pair handed back to the caller on a successful sign-in or refresh.</summary>
public sealed record AuthTokens(string AccessToken, string RefreshToken, int ExpiresIn);

/// <summary>
/// A newly issued pair plus the identifier of the persisted refresh token row, so a
/// rotation can point the superseded token at its replacement.
/// </summary>
public sealed record IssuedTokenPair(AuthTokens Tokens, Guid RefreshTokenId);

/// <summary>Outcome of a Google sign-in: either tokens, or a demand for OTP activation.</summary>
public sealed record GoogleSignInResult(bool RequiresActivation, AuthTokens? Tokens)
{
    public static GoogleSignInResult ActivationRequired() => new(true, null);

    public static GoogleSignInResult Success(AuthTokens tokens) => new(false, tokens);
}

public sealed record UserProfile(
    Guid Id,
    string Email,
    string? DisplayName,
    string Status,
    bool EmailConfirmed,
    IReadOnlyList<string> Roles,
    DateTimeOffset CreatedAt);

/// <summary>Identity data extracted from a validated Google ID token.</summary>
public sealed record GoogleUserInfo(string Subject, string Email, bool EmailVerified, string? Name);
