"""Synthetic acceptance inputs with closed-form references; no task/reference assets."""
SPICE = '''Voltage divider
V1 in 0 DC 10
R1 in out 1000
R2 out 0 1000
.control
op
wrdata divider.txt v(out)
.endc
.end
'''
HDL = '''module top;
  integer a,b;
  reg [1:0] x,y;
  wire [2:0] sum = {1'b0,x} + {1'b0,y};
  initial begin
    for (a=0;a<4;a=a+1) for (b=0;b<4;b=b+1) begin
      x=2'(a); y=2'(b); #1;
      if (sum !== a+b) $fatal(1,"sum mismatch");
      $display("SUM %0d %0d %0d",a,b,sum);
    end
    $finish;
  end
endmodule
'''
LIBERTY = '''library(tiny) {
 delay_model : table_lookup;
 time_unit : "1ns";
 voltage_unit : "1V";
 current_unit : "1mA";
 pulling_resistance_unit : "1kohm";
 capacitive_load_unit(1,pf);
 input_threshold_pct_rise : 50;
 input_threshold_pct_fall : 50;
 output_threshold_pct_rise : 50;
 output_threshold_pct_fall : 50;
 slew_lower_threshold_pct_rise : 20;
 slew_upper_threshold_pct_rise : 80;
 slew_lower_threshold_pct_fall : 20;
 slew_upper_threshold_pct_fall : 80;
 nom_process : 1;
 nom_voltage : 1;
 nom_temperature : 25;
 cell(BUF) {
  area : 1;
  pin(A) { direction : input; capacitance : 0.01; }
  pin(Y) { direction : output; function : "A";
   timing() { related_pin : "A"; timing_sense : positive_unate;
    cell_rise(scalar) { values("0.1"); }
    cell_fall(scalar) { values("0.1"); }
    rise_transition(scalar) { values("0.01"); }
    fall_transition(scalar) { values("0.01"); }
   }
  }
 }
}
'''
STA = '''read_liberty tiny.lib
read_verilog design.v
link_design top
create_clock -name clk -period 10
set_input_delay 1 -clock clk [get_ports a]
set_output_delay 1 -clock clk [get_ports y]
report_checks -path_delay max -format full_clock_expanded -digits 6
'''
INP = '''*HEADING
Uniform axial extension E=1000 A=1 F=4 L=1, nu=0
*NODE,NSET=NALL
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT,TYPE=C3D8,ELSET=SOLID
1,1,2,3,4,5,6,7,8
*NSET,NSET=LEFT
1,4,5,8
*NSET,NSET=RIGHT
2,3,6,7
*MATERIAL,NAME=ELASTIC
*ELASTIC
1000.,0.
*SOLID SECTION,ELSET=SOLID,MATERIAL=ELASTIC
*BOUNDARY
LEFT,1,1
NALL,2,3
*STEP
*STATIC
*CLOAD
RIGHT,1,1.
*NODE PRINT,NSET=RIGHT
U
*NODE FILE
U
*END STEP
'''
MOS = '''loadString("model Decay Real x(start=1, fixed=true); equation der(x)=-x; end Decay;");
simulate(Decay, startTime=0, stopTime=1, numberOfIntervals=10, tolerance=1e-8, outputFormat="csv");
'''
CASES = {'ngspice':(SPICE, {}), 'iverilog':(HDL, {}), 'verilator':(HDL, {}),
         'opensta':(STA, {'tiny.lib': LIBERTY, 'design.v': 'module top(input a, output y); BUF b(.A(a),.Y(y)); endmodule\n'}),
         'calculix':(INP, {}), 'openmodelica':(MOS, {})}
INVALID = {'ngspice': 'bad\nV1 in 0 garbage\n.op\n.end\n',
           'iverilog':'module top; syntax broken; endmodule',
           'verilator':'module top; syntax broken; endmodule',
           'opensta':'this_command_does_not_exist\n',
           'calculix':'*HEADING\ninvalid\n*STEP\n*STATIC\n*END STEP\n',
           'openmodelica':'loadString("model Broken equation x = ; end Broken;");\n'}
