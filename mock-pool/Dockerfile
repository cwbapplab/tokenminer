# Build the mock pool as a small, non-root runtime image.

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build

WORKDIR /src

COPY . .

RUN dotnet publish src/MockPool/MockPool.csproj -c Release -o /app

FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime

RUN useradd --create-home --shell /usr/sbin/nologin pool

WORKDIR /app

COPY --from=build /app .

USER pool

ENV MOCK_POOL_PRL_ADDR=0.0.0.0:3335
ENV MOCK_POOL_QTC_ADDR=0.0.0.0:3336
ENV MOCK_POOL_LOG_LEVEL=info

EXPOSE 3335 3336

ENTRYPOINT ["dotnet", "MockPool.dll"]
