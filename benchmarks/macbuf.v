// custom_macbuf: a hand-written design for this project's own zero-shot
// generalization test. Not derived from any existing benchmark suite.
//
// Structure: a 3-tap weighted multiply-accumulate pipeline reading from one
// input sample buffer and writing accumulated results into a second output
// buffer. Three independent 32-bit multiplies (DSP-inference) and two
// distinct memory arrays (BRAM-inference) -- a DSP/BRAM mix not matching
// any existing benchmark's exact composition (3 DSP, 2 BRAM).
module custom_macbuf (
    input clk,
    input reset,
    input wen,
    input [31:0] sample_in,
    input [4:0] waddr,
    input [4:0] raddr,
    output reg [31:0] result_out
);

    reg [31:0] sample_buf [0:31];   // BRAM #1: input sample ring buffer
    reg [31:0] result_buf [0:31];   // BRAM #2: accumulated result buffer

    reg [31:0] w0, w1, w2;
    reg [31:0] sample_r;

    wire [31:0] p0, p1, p2;
    wire [31:0] mac_sum;

    assign p0 = sample_r * w0;
    assign p1 = sample_r * w1;
    assign p2 = sample_r * w2;
    assign mac_sum = p0 + p1 + p2;

    always @(posedge clk) begin
        if (reset == 1'b1) begin
            w0 <= 32'd3;
            w1 <= 32'd5;
            w2 <= 32'd7;
            sample_r <= 32'd0;
            result_out <= 32'd0;
        end else begin
            if (wen)
                sample_buf[waddr] <= sample_in;

            sample_r <= sample_buf[raddr];
            result_out <= mac_sum;
            result_buf[raddr] <= mac_sum;
        end
    end

endmodule
