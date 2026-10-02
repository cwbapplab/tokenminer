using Microsoft.Extensions.Logging;
using Microsoft.Playwright;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Abstractions;

namespace TokenMiner.Infrastructure.Integrations;

/// <summary>
/// Places a pool withdrawal through the pool's web UI. Used only for pools that offer no way to
/// request a payout over an API.
/// </summary>
public interface IPoolWithdrawalClient
{
    Task<PoolWithdrawalResult> RequestAsync(
        PoolPayoutContext context,
        decimal amount,
        CancellationToken cancellationToken);
}

/// <summary>
/// Default client: refuses to act. Requesting a payout is a real financial action, so it stays
/// off until an operator explicitly enables it.
/// </summary>
internal sealed class DisabledPoolWithdrawalClient : IPoolWithdrawalClient
{
    public Task<PoolWithdrawalResult> RequestAsync(
        PoolPayoutContext context,
        decimal amount,
        CancellationToken cancellationToken) =>
        Task.FromResult(new PoolWithdrawalResult(
            Requested: false,
            TransactionHash: null,
            Detail: "Browser withdrawals are disabled (Treasury:EnableBrowserWithdrawals). "
                + "Withdraw manually or enable the automation."));
}

/// <summary>
/// Playwright-driven withdrawal. Requires the pool to be reachable with stored credentials and
/// the Playwright browsers to be installed (<c>playwright install chromium</c>).
/// </summary>
/// <remarks>
/// The automation is intentionally minimal and defensive: it never guesses at a selector that is
/// missing, and it reports back rather than retrying blindly, because a duplicated payout
/// request is worse than a missed one.
/// </remarks>
internal sealed class BrowserPoolWithdrawalClient(
    TreasuryOptions options,
    ILogger<BrowserPoolWithdrawalClient> logger) : IPoolWithdrawalClient
{
    public async Task<PoolWithdrawalResult> RequestAsync(
        PoolPayoutContext context,
        decimal amount,
        CancellationToken cancellationToken)
    {
        if (!options.EnableBrowserWithdrawals)
        {
            return new PoolWithdrawalResult(false, null, "Browser withdrawals are disabled.");
        }

        if (string.IsNullOrWhiteSpace(options.WithdrawalPageUrl) || string.IsNullOrWhiteSpace(options.WithdrawalAddress))
        {
            return new PoolWithdrawalResult(
                false,
                null,
                "Treasury:WithdrawalPageUrl and Treasury:WithdrawalAddress must be configured.");
        }

        try
        {
            using var playwright = await Playwright.CreateAsync();

            await using var browser = await playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
            {
                Headless = true,
            });

            var page = await browser.NewPageAsync();

            await page.GotoAsync(
                $"{options.WithdrawalPageUrl}?coin={Uri.EscapeDataString(context.CoinCode)}",
                new PageGotoOptions { Timeout = options.BrowserTimeoutSeconds * 1000 });

            // The operator's previously authenticated session is expected to be present via the
            // storage state file; without it the page will redirect to the sign-in form.
            var amountField = page.Locator(options.WithdrawalAmountSelector);
            if (await amountField.CountAsync() == 0)
            {
                return new PoolWithdrawalResult(
                    false,
                    null,
                    $"Withdrawal form not found on {options.WithdrawalPageUrl}. Is the session still authenticated?");
            }

            await amountField.First.FillAsync(amount.ToString(System.Globalization.CultureInfo.InvariantCulture));

            var addressField = page.Locator(options.WithdrawalAddressSelector);
            if (await addressField.CountAsync() > 0)
            {
                await addressField.First.FillAsync(options.WithdrawalAddress);
            }

            var submit = page.Locator(options.WithdrawalSubmitSelector);
            if (await submit.CountAsync() == 0)
            {
                return new PoolWithdrawalResult(false, null, "Withdrawal submit control not found.");
            }

            await submit.First.ClickAsync();

            logger.LogInformation(
                "Submitted a browser withdrawal of {Amount} {Coin} for pool {Pool}.",
                amount,
                context.CoinCode,
                context.SystemPoolId);

            // The platform is the source of truth for the resulting transaction; the payout is
            // picked up from the payout history on a later monitor run.
            return new PoolWithdrawalResult(true, null, "Withdrawal submitted through the pool UI.");
        }
        catch (Exception exception)
        {
            logger.LogError(exception, "A browser withdrawal for {Pool} failed.", context.SystemPoolId);

            return new PoolWithdrawalResult(false, null, $"Browser withdrawal failed: {exception.Message}");
        }
    }
}
