import * as sim from "lib-simulation-wasm";

const simulation = new sim.Simulation();
const world = simulation.world();

const viewport = document.getElementById("viewport");

const width = viewport.width;
const height = viewport.height;

const scale = window.devicePixelRatio || 1;

viewport.width = width * scale;
viewport.height = height * scale;

viewport.style.width = width + "px";
viewport.style.height = height + "px";

const ctx = viewport.getContext("2d");
ctx.scale(scale, scale);

CanvasRenderingContext2D.prototype.drawTriangle = function (
  x,
  y,
  rotation,
  size
) {
  this.beginPath();
  this.moveTo(x - Math.sin(rotation), y + Math.cos(rotation));
  this.lineTo(
    x - Math.sin(rotation + (2.0 / 3.0) * Math.PI) * size,
    y + Math.cos(rotation + (2.0 / 3.0) * Math.PI) * size
  );
  this.lineTo(
    x - Math.sin(rotation + (4.0 / 3.0) * Math.PI) * size,
    y + Math.cos(rotation + (4.0 / 3.0) * Math.PI) * size
  );
  this.lineTo(x - Math.sin(rotation) * size, y + Math.cos(rotation) * size);

  this.stroke();
};

for (const animal of simulation.world().animals) {
  ctx.drawTriangle(
    animal.x * width,
    animal.y * height,
    animal.rotation,
    0.01 * width
  );
}
