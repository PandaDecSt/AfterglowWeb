// AfterglowRender Terrain Hydrology Shader
// Ported from TerrainCommon.hlsl, TerrainData_CS.hlsl

// ============================================================
// Terrain Constants
// ============================================================

const TERRAIN_MESH_INTERVAL: f32 = 4.0;
const TERRAIN_MESH_SIDE_ELEMENTS: u32 = 256u;
const TERRAIN_DATA_SIDE_ELEMENTS: u32 = 4096u;

const TERRAIN_TEX_COORD_SCALING: f32 = 0.05;

const WATER_MESH_INTERVAL: f32 = 4.0;
const WATER_MESH_SIDE_ELEMENTS: u32 = 256u;
const WATER_DATA_SIDE_ELEMENTS: u32 = 4096u;

const WATER_BASE_HEIGHT: f32 = 0.0;
const WATER_SEDIMENT_CAPABILITY: f32 = 0.2;
const WATER_INV_SEDIMENT_CAPABILITY: f32 = 5.0;

const TERRAIN_DATA_INTERVAL: f32 = 0.024414062; // worldSideLength / terrainDataSideElements
const WATER_DATA_INTERVAL: f32 = 0.024414062;

const TERRAIN_DATA_CELL_AREA: f32 = 0.000596046; // terrainDataInterval^2

// ============================================================
// Water Flow Simulation
// ============================================================

struct WaterCell {
  terrainHeight: f32,
  waterHeight: f32,
  waterVelocityX: f32,
  waterVelocityZ: f32,
  sediment: f32,
  rainfall: f32,
  temperature: f32,
  humidity: f32,
};

// Calculate water flow between cells
fn CalculateWaterFlow(
  height: vec2<f32>,      // (terrain, water) at current cell
  heightR: vec2<f32>,     // (terrain, water) at right neighbor
  heightT: vec2<f32>,     // (terrain, water) at top neighbor
  heightL: vec2<f32>,     // (terrain, water) at left neighbor
  heightB: vec2<f32>,     // (terrain, water) at bottom neighbor
  deltaTime: f32
) -> vec4<f32> {
  // Total heights
  let totalHeight = height.x + height.y;
  let totalHeightR = heightR.x + heightR.y;
  let totalHeightT = heightT.x + heightT.y;
  let totalHeightL = heightL.x + heightL.y;
  let totalHeightB = heightB.x + heightB.y;

  // Height differences (water flows from high to low)
  let dR = totalHeight - totalHeightR;
  let dT = totalHeight - totalHeightT;
  let dL = totalHeight - totalHeightL;
  let dB = totalHeight - totalHeightB;

  // Simple flow model: flow rate proportional to height difference
  let flowRate = 0.1;
  let maxFlow = height.y * 0.25; // Can't flow more than 25% of water per step

  var flow = vec4<f32>(0.0);
  flow.x = clamp(dR * flowRate, -maxFlow, maxFlow); // Right
  flow.y = clamp(dT * flowRate, -maxFlow, maxFlow); // Top
  flow.z = clamp(dL * flowRate, -maxFlow, maxFlow); // Left
  flow.w = clamp(dB * flowRate, -maxFlow, maxFlow); // Bottom

  // Normalize flows
  let totalOut = max(flow.x, 0.0) + max(flow.y, 0.0) + max(flow.z, 0.0) + max(flow.w, 0.0);
  let totalIn = max(-flow.x, 0.0) + max(-flow.y, 0.0) + max(-flow.z, 0.0) + max(-flow.w, 0.0);
  let scaleFactor = 1.0 / max(totalOut, totalIn);

  return flow * scaleFactor;
}

// ============================================================
// Terrain Surface Variants (from TerrainCommon.hlsl)
// ============================================================

// Calculate terrain normal from height data
fn CalculateTerrainNormal(
  height: vec2<f32>,      // (terrain, water) at current
  heightR: vec2<f32>,     // (terrain, water) at right
  heightT: vec2<f32>      // (terrain, water) at top
) -> vec4<f32> {
  let deltaTerrainHeight = vec4<f32>(
    height.x - heightR.x,
    height.x - heightT.x,
    height.y - heightR.y,
    height.y - heightT.y
  ) * (1.0 / TERRAIN_DATA_INTERVAL);

  return vec4<f32>(
    normalize(vec3<f32>(deltaTerrainHeight.xy, 1.0)).xy,
    normalize(vec3<f32>(deltaTerrainHeight.zw, 1.0)).xy
  );
}

// Variant terrain surface color based on temperature/humidity
fn VariantTerrainSurface(
  srcColor: vec3<f32>,
  terrainSurface: vec4<f32>,
  temperature: f32
) -> vec3<f32> {
  var outColor = vec3<f32>(0.0);

  // Beach (terrainSurface.g)
  outColor = mix(
    srcColor,
    Desaturation(HueShift(srcColor, 0.95), 0.5) * 4.0,
    terrainSurface.g
  );

  // Humidity (terrainSurface.b)
  outColor = mix(
    outColor,
    Desaturation(HueShift(outColor, 0.95), -0.35) * 4.0,
    terrainSurface.b
  );

  // Snow (based on temperature)
  outColor = mix(outColor, vec3<f32>(0.7, 0.7, 0.75), saturate(-temperature * 0.5));

  return outColor;
}

// ============================================================
// Water Surface Rendering
// ============================================================

// Calculate water normal from adjacent water heights
fn CalculateWaterNormal(
  waterHeight: f32,
  waterHeightR: f32,
  waterHeightT: f32
) -> vec3<f32> {
  let dx = waterHeight - waterHeightR;
  let dy = waterHeight - waterHeightT;
  return normalize(vec3<f32>(dx, dy, 1.0));
}

// Water surface color with Fresnel and depth
fn WaterSurfaceColor(
  waterNormal: vec3<f32>,
  viewDir: vec3<f32>,
  lightDir: vec3<f32>,
  waterDepth: f32,
  waterColor: vec3<f32>
) -> vec3<f32> {
  // Fresnel
  let NdotV = max(dot(waterNormal, viewDir), 0.0);
  let fresnel = pow(1.0 - NdotV, 5.0) * 0.04 + 0.96;

  // Reflection
  let reflection = reflect(-viewDir, waterNormal);
  let skyColor = mix(vec3<f32>(0.3, 0.5, 0.8), vec3<f32>(0.1, 0.2, 0.4), reflection.y * 0.5 + 0.5);

  // Refraction (simplified)
  let refraction = mix(waterColor, vec3<f32>(0.1, 0.3, 0.5), saturate(waterDepth * 0.1));

  // Specular highlight
  let H = normalize(viewDir + lightDir);
  let spec = pow(max(dot(waterNormal, H), 0.0), 256.0);

  // Final color
  var color = mix(refraction, skyColor, fresnel);
  color += vec3<f32>(1.0) * spec * 0.8;

  return color;
}

// ============================================================
// Rainfall System
// ============================================================

struct RainfallConfig {
  intensity: f32,        // mm/hour
  evaporationRate: f32,  // mm/hour
  sedimentCapacity: f32, // max sediment per cell
  temperature: f32,      // affects evaporation
  humidity: f32,         // affects evaporation
};

// Update rainfall for a cell
fn UpdateRainfall(
  cell: WaterCell,
  config: RainfallConfig,
  deltaTime: f32
) -> WaterCell {
  var updated = cell;

  // Evaporation (higher when hot, lower when humid)
  let evapFactor = config.temperature * (1.0 - config.humidity);
  let evaporation = config.evaporationRate * evapFactor * deltaTime;

  // Rainfall accumulation
  let rainfallDelta = config.intensity * deltaTime;

  // Update water height
  updated.waterHeight = max(0.0, updated.waterHeight + rainfallDelta - evaporation);

  // Update humidity (more humidity when water is present)
  updated.humidity = saturate(updated.humidity + updated.waterHeight * 0.01 - evaporation * 0.1);

  // Update temperature (cooler when wet)
  updated.temperature = mix(updated.temperature, updated.temperature * 0.99, updated.waterHeight * 0.1);

  return updated;
}

// ============================================================
// Sediment Transport
// ============================================================

// Calculate sediment transport capacity
fn SedimentCapacity(waterVelocity: f32, waterDepth: f32) -> f32 {
  return WATER_SEDIMENT_CAPABILITY * waterVelocity * waterDepth;
}

// Erode or deposit sediment
fn UpdateSediment(
  cell: WaterCell,
  sedimentCapacity: f32,
  deltaTime: f32
) -> WaterCell {
  var updated = cell;

  if (updated.sediment < sedimentCapacity) {
    // Erode terrain
    let erosionRate = (sedimentCapacity - updated.sediment) * 0.1 * deltaTime;
    updated.terrainHeight = max(0.0, updated.terrainHeight - erosionRate);
    updated.sediment += erosionRate;
  } else {
    // Deposit sediment
    let depositRate = (updated.sediment - sedimentCapacity) * 0.1 * deltaTime;
    updated.terrainHeight += depositRate;
    updated.sediment -= depositRate;
  }

  updated.sediment = max(0.0, updated.sediment);

  return updated;
}

// ============================================================
// Utility Functions (duplicated from toon-textured for standalone)
// ============================================================

fn Desaturation(color: vec3<f32>, amount: f32) -> vec3<f32> {
  let grey = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  return mix(color, vec3<f32>(grey), amount);
}

fn HueShift(color: vec3<f32>, amount: f32) -> vec3<f32> {
  // Simplified hue shift
  let angle = amount * 3.14159265;
  let s = sin(angle);
  let c = cos(angle);
  return vec3<f32>(
    color.r * (0.787 * c + 0.213) + color.g * (-0.213 * c + 0.213 * s) + color.b * (0.143 * c - 0.213 * s),
    color.r * (-0.213 * c + 0.143 * s) + color.g * (0.787 * c + 0.213) + color.b * (-0.143 * c + 0.213 * s),
    color.r * (0.143 * c + 0.213 * s) + color.g * (-0.213 * c + 0.787 * s) + color.b * (0.787 * c + 0.213)
  );
}
