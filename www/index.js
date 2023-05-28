import * as sim from "lib-simulation-wasm";

const simulation = new sim.Simulation();

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
  this.moveTo(
    x - Math.sin(rotation) * size * 1.5,
    y + Math.cos(rotation) * size * 1.5
  );
  this.lineTo(
    x - Math.sin(rotation + (2.0 / 3.0) * Math.PI) * size,
    y + Math.cos(rotation + (2.0 / 3.0) * Math.PI) * size
  );
  this.lineTo(
    x - Math.sin(rotation + (4.0 / 3.0) * Math.PI) * size,
    y + Math.cos(rotation + (4.0 / 3.0) * Math.PI) * size
  );
  this.lineTo(
    x - Math.sin(rotation) * size * 1.5,
    y + Math.cos(rotation) * size * 1.5
  );
  this.strokeStyle = "white";
  this.stroke();
  this.fillStyle = "#434a6d";
  this.fill();
};

CanvasRenderingContext2D.prototype.drawCircle = function (
  x,
  y,
  radius
) {
  this.beginPath();
  this.arc(x, y, radius, 0, 2 * Math.PI);
  this.fillStyle = "#00ff80";
  this.fill();
};

function redraw() {
  ctx.clearRect(0, 0, width, height);

  simulation.step();

  const world = simulation.world();

  for (const food of world.foods) {
    ctx.drawCircle(food.x * width, food.y * height, 0.005 * width);
  }

  for (const animal of simulation.world().animals) {
    ctx.drawTriangle(
      animal.x * width,
      animal.y * height,
      animal.rotation,
      0.01 * width
    );
  }

  requestAnimationFrame(redraw);
}

redraw();
