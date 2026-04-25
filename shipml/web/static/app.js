const palette = [
  "#0f766e",
  "#2563eb",
  "#9333ea",
  "#dc2626",
  "#ca8a04",
  "#0891b2",
  "#4d7c0f",
  "#be185d",
  "#7c3aed",
  "#ea580c",
  "#155e75",
  "#475569",
];

const state = {
  summary: null,
  routeLayer: null,
  trackLayer: null,
  shipLayer: null,
};

const map = L.map("map", {
  zoomControl: false,
  preferCanvas: true,
}).setView([56.0, 10.5], 6);

L.control.zoom({ position: "bottomleft" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const shipTypeSelect = document.getElementById("shipTypeSelect");
const routeSelect = document.getElementById("routeSelect");
const anomalyOnly = document.getElementById("anomalyOnly");
const trackToggle = document.getElementById("trackToggle");
const shipCount = document.getElementById("shipCount");
const shownShipCount = document.getElementById("shownShipCount");
const modelName = document.getElementById("modelName");
const modelAccuracy = document.getElementById("modelAccuracy");
const routeList = document.getElementById("routeList");

async function init() {
  const response = await fetch("/api/summary");
  state.summary = await response.json();
  populateFilters(state.summary);
  setModelSummary(state.summary.model);
  await refreshMap();

  shipTypeSelect.addEventListener("change", refreshMap);
  routeSelect.addEventListener("change", refreshMap);
  anomalyOnly.addEventListener("change", refreshMap);
  trackToggle.addEventListener("change", refreshMap);
}

function populateFilters(summary) {
  shipTypeSelect.innerHTML = "";
  const allOption = new Option("전체", "__all__");
  shipTypeSelect.appendChild(allOption);
  summary.shipTypes.forEach((item) => {
    shipTypeSelect.appendChild(new Option(`${item.name} (${item.count})`, item.name));
  });
  if (summary.shipTypes.length > 0) {
    shipTypeSelect.value = summary.shipTypes[0].name;
  }

  routeSelect.innerHTML = "";
  routeSelect.appendChild(new Option("전체", "__all__"));
  summary.routes.forEach((item) => {
    routeSelect.appendChild(new Option(`${item.name} (${item.count})`, item.name));
  });
}

function setModelSummary(model) {
  modelName.textContent = model?.displayName || "-";
  modelAccuracy.textContent =
    typeof model?.accuracy === "number" ? `${(model.accuracy * 100).toFixed(1)}%` : "-";
}

async function refreshMap() {
  const params = new URLSearchParams({
    ship_type: shipTypeSelect.value || "__all__",
    route: routeSelect.value || "__all__",
    anomaly: anomalyOnly.checked ? "1" : "0",
    tracks: trackToggle.checked ? "1" : "0",
    max_ships: "800",
  });
  const response = await fetch(`/api/map-data?${params}`);
  const data = await response.json();
  renderMap(data);
  renderSummary(data);
}

function renderMap(data) {
  [state.routeLayer, state.trackLayer, state.shipLayer].forEach((layer) => {
    if (layer) {
      map.removeLayer(layer);
    }
  });

  state.trackLayer = L.geoJSON(data.shipTracks, {
    style: (feature) => ({
      color: routeColor(feature.properties.route),
      weight: feature.properties.is_anomaly ? 1.6 : 0.8,
      opacity: feature.properties.is_anomaly ? 0.55 : 0.22,
      dashArray: "4 6",
    }),
  }).addTo(map);

  state.routeLayer = L.geoJSON(data.routes, {
    style: (feature) => ({
      color: routeColor(feature.properties.route_label),
      weight: Math.min(8, 2.5 + Math.log2((feature.properties.vessel_count || 1) + 1)),
      opacity: 0.9,
    }),
    onEachFeature: (feature, layer) => {
      layer.bindPopup(routePopup(feature.properties));
    },
  }).addTo(map);

  state.shipLayer = L.geoJSON(data.ships, {
    pointToLayer: (feature, latlng) => {
      return L.marker(latlng, {
        icon: shipIcon(feature.properties),
        riseOnHover: true,
      });
    },
    onEachFeature: (feature, layer) => {
      layer.bindPopup(shipPopup(feature.properties));
    },
  }).addTo(map);

  const bounds = combinedBounds([state.routeLayer, state.trackLayer, state.shipLayer]);
  if (bounds?.isValid()) {
    map.fitBounds(bounds.pad(0.12), { animate: true, maxZoom: 10 });
  } else if (data.bounds) {
    map.fitBounds(data.bounds, { animate: true, maxZoom: 10 });
  }
}

function renderSummary(data) {
  shipCount.textContent = numberText(data.shipCount);
  shownShipCount.textContent = numberText(data.shownShipCount);
  routeList.innerHTML = "";
  data.routeSummary.forEach((item) => {
    const row = document.createElement("div");
    row.className = "route-row";
    const color = routeColor(item.route_label);
    row.innerHTML = `
      <span class="route-swatch" style="background:${color}"></span>
      <span class="route-name">${escapeHtml(item.route_label)}</span>
      <span class="route-count">${numberText(item.vessel_count)}</span>
    `;
    routeList.appendChild(row);
  });
}

function shipIcon(properties) {
  const angle = Number.isFinite(properties.bearing) ? properties.bearing : 0;
  const anomalyClass = properties.is_anomaly ? " is-anomaly" : "";
  return L.divIcon({
    className: "ship-marker",
    html: `<div class="ship-arrow${anomalyClass}" style="--angle:${angle}deg"></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

function shipPopup(properties) {
  return `
    <p class="popup-title">${escapeHtml(properties.mmsi)}</p>
    <div class="popup-grid">
      <span>선종</span><strong>${escapeHtml(properties.shiptype)}</strong>
      <span>항로</span><strong>${escapeHtml(properties.route)}</strong>
      <span>속도</span><strong>${formatValue(properties.mean_sog, "kn")}</strong>
      <span>선종 확률</span><strong>${formatPercent(properties.shiptype_probability)}</strong>
      <span>항로 확률</span><strong>${formatPercent(properties.route_probability)}</strong>
      <span>이상</span><strong>${properties.is_anomaly ? "Y" : "N"}</strong>
    </div>
  `;
}

function routePopup(properties) {
  return `
    <p class="popup-title">${escapeHtml(properties.route_label)}</p>
    <div class="popup-grid">
      <span>선박</span><strong>${numberText(properties.vessel_count)}</strong>
      <span>이상</span><strong>${numberText(properties.anomaly_count)}</strong>
      <span>항로 확률</span><strong>${formatPercent(properties.avg_route_probability)}</strong>
      <span>선종 확률</span><strong>${formatPercent(properties.avg_shiptype_probability)}</strong>
    </div>
  `;
}

function combinedBounds(layers) {
  let bounds = null;
  layers.forEach((layer) => {
    if (!layer) {
      return;
    }
    const layerBounds = layer.getBounds?.();
    if (!layerBounds?.isValid()) {
      return;
    }
    bounds = bounds ? bounds.extend(layerBounds) : layerBounds;
  });
  return bounds;
}

function routeColor(label) {
  const text = String(label || "");
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return palette[hash % palette.length];
}

function numberText(value) {
  const number = Number(value || 0);
  return new Intl.NumberFormat("ko-KR").format(number);
}

function formatPercent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
}

function formatValue(value, suffix) {
  return typeof value === "number" ? `${value.toFixed(1)} ${suffix}` : "-";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init().catch((error) => {
  console.error(error);
  routeList.innerHTML = `<div class="route-row">지도 데이터를 불러오지 못했습니다.</div>`;
});
