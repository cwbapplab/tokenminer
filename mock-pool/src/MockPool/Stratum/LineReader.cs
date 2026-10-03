using System.Text;

namespace MockPool.Stratum;

/// <summary>Reads newline-delimited messages from a stream, tolerating a trailing CR and capping line size.</summary>
internal sealed class LineReader(Stream stream)
{
    // Pearl submits carry a base64 PlainProof of hundreds of KB; the spec allows a 4 MB frame.
    private const int MaxLineBytes = 4 * 1024 * 1024;

    private byte[] _buffer = new byte[8192];
    private int _length;

    public async ValueTask<string?> ReadLineAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            var newline = Array.IndexOf(_buffer, (byte)'\n', 0, _length);
            if (newline >= 0)
            {
                var line = Encoding.UTF8.GetString(_buffer, 0, newline).TrimEnd('\r');
                var remaining = _length - newline - 1;
                if (remaining > 0)
                {
                    Array.Copy(_buffer, newline + 1, _buffer, 0, remaining);
                }

                _length = remaining;
                return line;
            }

            if (_length == _buffer.Length)
            {
                if (_buffer.Length >= MaxLineBytes)
                {
                    throw new InvalidDataException($"Stratum line exceeded {MaxLineBytes} bytes.");
                }

                Array.Resize(ref _buffer, Math.Min(_buffer.Length * 2, MaxLineBytes));
            }

            var read = await stream.ReadAsync(_buffer.AsMemory(_length), cancellationToken).ConfigureAwait(false);
            if (read == 0)
            {
                if (_length == 0)
                {
                    return null;
                }

                var tail = Encoding.UTF8.GetString(_buffer, 0, _length).TrimEnd('\r');
                _length = 0;
                return tail;
            }

            _length += read;
        }
    }
}
