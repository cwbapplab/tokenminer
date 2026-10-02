namespace TokenMiner.Domain.Mining.Enums;

/// <summary>
/// How a pool delivers mined funds. Determines whether the pool monitor has to request a
/// withdrawal or merely observe payouts.
/// </summary>
public enum PayoutMode
{
    /// <summary>The pool accrues a balance we withdraw ourselves.</summary>
    ManagedRequest = 0,

    /// <summary>The pool pays out on its own threshold or schedule.</summary>
    NativeAutoPayout = 1,

    /// <summary>Mining rewards are sent straight to one of our own wallets.</summary>
    DirectToWallet = 2,
}

/// <summary>snake_case database representation, e.g. <c>native_auto_payout</c>.</summary>
public static class PayoutModeDbValues
{
    public const string ManagedRequest = "managed_request";
    public const string NativeAutoPayout = "native_auto_payout";
    public const string DirectToWallet = "direct_to_wallet";

    public static string ToDbValue(this PayoutMode mode) => mode switch
    {
        PayoutMode.ManagedRequest => ManagedRequest,
        PayoutMode.NativeAutoPayout => NativeAutoPayout,
        PayoutMode.DirectToWallet => DirectToWallet,
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown payout mode."),
    };

    public static PayoutMode FromDbValue(string value) => value switch
    {
        ManagedRequest => PayoutMode.ManagedRequest,
        NativeAutoPayout => PayoutMode.NativeAutoPayout,
        DirectToWallet => PayoutMode.DirectToWallet,
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown payout mode value."),
    };

    /// <summary>Non-throwing parse, for validating wire input.</summary>
    public static bool TryParseDbValue(string? value, out PayoutMode mode)
    {
        switch (value)
        {
            case ManagedRequest:
                mode = PayoutMode.ManagedRequest;
                return true;
            case NativeAutoPayout:
                mode = PayoutMode.NativeAutoPayout;
                return true;
            case DirectToWallet:
                mode = PayoutMode.DirectToWallet;
                return true;
            default:
                mode = PayoutMode.ManagedRequest;
                return false;
        }
    }
}
