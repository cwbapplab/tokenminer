using MediatR;

namespace TokenMiner.Application.Treasury.Commands;

public sealed record ConversionPipelineResult(int Queued, int Advanced);

public sealed record RunConversionPipelineCommand(int BatchSize) : IRequest<ConversionPipelineResult>;

/// <summary>
/// One pass of the conversion pipeline: queue conversions for newly settled payouts, then drive
/// the in-flight ones. Splitting the work into a single entry point keeps the job trivial.
/// </summary>
internal sealed class RunConversionPipelineCommandHandler(ISender sender)
    : IRequestHandler<RunConversionPipelineCommand, ConversionPipelineResult>
{
    public async Task<ConversionPipelineResult> Handle(
        RunConversionPipelineCommand request,
        CancellationToken cancellationToken)
    {
        var queued = await sender.Send(new QueuePayoutConversionsCommand(), cancellationToken);
        var advanced = await sender.Send(new AdvanceConversionsCommand(request.BatchSize), cancellationToken);

        return new ConversionPipelineResult(queued, advanced);
    }
}
