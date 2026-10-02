namespace TokenMiner.Domain.Users.Enums;

public enum OtpPurpose
{
    Activation = 0,
    EmailChange = 1,
    Login = 2,
}

/// <summary>snake_case database representation, e.g. <c>email_change</c>.</summary>
public static class OtpPurposeDbValues
{
    public const string Activation = "activation";
    public const string EmailChange = "email_change";
    public const string Login = "login";

    public static string ToDbValue(this OtpPurpose purpose) => purpose switch
    {
        OtpPurpose.Activation => Activation,
        OtpPurpose.EmailChange => EmailChange,
        OtpPurpose.Login => Login,
        _ => throw new ArgumentOutOfRangeException(nameof(purpose), purpose, "Unknown OTP purpose."),
    };

    public static OtpPurpose FromDbValue(string value) => value switch
    {
        Activation => OtpPurpose.Activation,
        EmailChange => OtpPurpose.EmailChange,
        Login => OtpPurpose.Login,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown OTP purpose value."),
    };
}
