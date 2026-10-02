namespace TokenMiner.Contracts.Auth;

public sealed record RegisterRequest(string Email, string Password, string? DisplayName);

public sealed record VerifyEmailRequest(string Token);

public sealed record RequestOtpRequest(string Email);

public sealed record VerifyOtpRequest(string Email, string Code);

public sealed record GoogleSignInRequest(string IdToken);

public sealed record LoginRequest(string Email, string Password);

public sealed record RefreshTokenRequest(string RefreshToken);

public sealed record LogoutRequest(string RefreshToken);
