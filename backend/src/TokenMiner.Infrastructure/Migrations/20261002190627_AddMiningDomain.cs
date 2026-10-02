using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddMiningDomain : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "coins",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    code = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    name = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    network = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    decimals = table.Column<int>(type: "integer", nullable: false),
                    last_known_usd_value = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    last_value_date = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_coins", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "mining_algos",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    code = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    name = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    priority = table.Column<int>(type: "integer", nullable: false),
                    configuration = table.Column<string>(type: "jsonb", nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_mining_algos", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "pools",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    system_pool_id = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    name = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_pools", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "user_hardware",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_id = table.Column<Guid>(type: "uuid", nullable: false),
                    hardware_id = table.Column<Guid>(type: "uuid", nullable: false),
                    name = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    last_seen_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_user_hardware", x => x.id);
                    table.ForeignKey(
                        name: "fk_user_hardware_users_user_id",
                        column: x => x.user_id,
                        principalTable: "users",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "coin_price_history",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    usd_value = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    observed_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    source = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_coin_price_history", x => x.id);
                    table.ForeignKey(
                        name: "fk_coin_price_history_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "pool_coins",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    pool_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_pool_coins", x => x.id);
                    table.ForeignKey(
                        name: "fk_pool_coins_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_pool_coins_pools_pool_id",
                        column: x => x.pool_id,
                        principalTable: "pools",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "pool_details",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    pool_id = table.Column<Guid>(type: "uuid", nullable: false),
                    base_url = table.Column<string>(type: "character varying(512)", maxLength: 512, nullable: false),
                    status_endpoint = table.Column<string>(type: "character varying(512)", maxLength: 512, nullable: true),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    payout_mode = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    payout_address = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    payout_network = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_pool_details", x => x.id);
                    table.ForeignKey(
                        name: "fk_pool_details_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_pool_details_pools_pool_id",
                        column: x => x.pool_id,
                        principalTable: "pools",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "ix_coin_price_history_coin_id_observed_at",
                table: "coin_price_history",
                columns: new[] { "coin_id", "observed_at" });

            migrationBuilder.CreateIndex(
                name: "ix_coins_code",
                table: "coins",
                column: "code",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_mining_algos_code",
                table: "mining_algos",
                column: "code",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_mining_algos_priority",
                table: "mining_algos",
                column: "priority");

            migrationBuilder.CreateIndex(
                name: "ix_pool_coins_coin_id",
                table: "pool_coins",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_pool_coins_pool_id_coin_id",
                table: "pool_coins",
                columns: new[] { "pool_id", "coin_id" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_pool_details_coin_id",
                table: "pool_details",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_pool_details_pool_id",
                table: "pool_details",
                column: "pool_id",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_pools_system_pool_id",
                table: "pools",
                column: "system_pool_id",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_user_hardware_user_id_hardware_id",
                table: "user_hardware",
                columns: new[] { "user_id", "hardware_id" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "coin_price_history");

            migrationBuilder.DropTable(
                name: "mining_algos");

            migrationBuilder.DropTable(
                name: "pool_coins");

            migrationBuilder.DropTable(
                name: "pool_details");

            migrationBuilder.DropTable(
                name: "user_hardware");

            migrationBuilder.DropTable(
                name: "coins");

            migrationBuilder.DropTable(
                name: "pools");
        }
    }
}
