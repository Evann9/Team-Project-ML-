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
  performance: null,
  routeLayer: null,
  trackLayer: null,
  futureLayer: null,
  shipLayer: null,
};

const map = L.map("map", {
  zoomControl: false,
  preferCanvas: true,
}).setView([56.0, 10.5], 6);

L.control.zoom({ position: "topright" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const shipTypeSelect = document.getElementById("shipTypeSelect");
const routeSelect = document.getElementById("routeSelect");
const anomalyOnly = document.getElementById("anomalyOnly");
const trackToggle = document.getElementById("trackToggle");
const futureToggle = document.getElementById("futureToggle");
const shipCount = document.getElementById("shipCount");
const shownShipCount = document.getElementById("shownShipCount");
const modelName = document.getElementById("modelName");
const modelAccuracy = document.getElementById("modelAccuracy");
const modelMacroF1 = document.getElementById("modelMacroF1");
const modelMethod = document.getElementById("modelMethod");
const modelNote = document.getElementById("modelNote");
const futureModelNote = document.getElementById("futureModelNote");
const routeAccuracy = document.getElementById("routeAccuracy");
const routeMacroF1 = document.getElementById("routeMacroF1");
const routeStrictAccuracy = document.getElementById("routeStrictAccuracy");
const routeTemporalAccuracy = document.getElementById("routeTemporalAccuracy");
const routeModelNote = document.getElementById("routeModelNote");
const futureBaselineNote = document.getElementById("futureBaselineNote");
const routeList = document.getElementById("routeList");
const performanceList = document.getElementById("performanceList");
const confusionList = document.getElementById("confusionList");

async function init() {
  const [summaryResponse, performanceResponse] = await Promise.all([
    fetch("/api/summary"),
    fetch("/api/model-performance"),
  ]);
  state.summary = await summaryResponse.json();
  state.performance = await performanceResponse.json();
  populateFilters(state.summary);
  setModelSummary(state.summary.model);
  setRouteModelSummary(state.summary.routeModel);
  setFutureModelSummary(state.summary.futureModel);
  renderPerformance(state.performance);
  await refreshMap();

  shipTypeSelect.addEventListener("change", refreshMap);
  routeSelect.addEventListener("change", refreshMap);
  anomalyOnly.addEventListener("change", refreshMap);
  trackToggle.addEventListener("change", refreshMap);
  futureToggle.addEventListener("change", refreshMap);
}

function populateFilters(summary) {
  shipTypeSelect.innerHTML = "";
  const allOption = new Option("전체 선종", "__all__");
  shipTypeSelect.appendChild(allOption);
  summary.shipTypes.forEach((item) => {
    shipTypeSelect.appendChild(new Option(`${item.name} (${item.count})`, item.name));
  });
  if (summary.shipTypes.length > 0) {
    shipTypeSelect.value = summary.shipTypes[0].name;
  }

  routeSelect.innerHTML = "";
  routeSelect.appendChild(new Option("전체 항로", "__all__"));
  summary.routes.forEach((item) => {
    routeSelect.appendChild(new Option(`${item.name} (${item.count})`, item.name));
  });
}

function setModelSummary(model) {
  modelName.textContent = model?.displayName || "-";
  modelAccuracy.textContent =
    typeof model?.accuracy === "number" ? `${(model.accuracy * 100).toFixed(1)}%` : "-";
  modelMacroF1.textContent =
    typeof model?.macroF1 === "number" ? `${(model.macroF1 * 100).toFixed(1)}%` : "-";
  modelMethod.textContent = methodLabel(model?.evaluationMethod);
  const overlap =
    typeof model?.groupOverlap === "number" ? `MMSI 중복 ${numberText(model.groupOverlap)}건` : "";
  const rows =
    typeof model?.testRows === "number" ? `테스트 ${numberText(model.testRows)}행` : "";
  modelNote.textContent = [overlap, rows].filter(Boolean).join(" · ") || "평가 정보 없음";
}

function setRouteModelSummary(model) {
  routeAccuracy.textContent = formatPercent(model?.accuracy);
  routeMacroF1.textContent = formatPercent(model?.macroF1);
  routeStrictAccuracy.textContent = formatPercent(model?.strictAccuracy);
  routeTemporalAccuracy.textContent = formatPercent(model?.temporalAccuracy);

  if (!model?.available) {
    routeModelNote.textContent = "항로 검증 정보 없음";
    return;
  }
  const earlyWindow =
    typeof model.earlyWindowHours === "number" ? `초기 ${model.earlyWindowHours}h` : "초기 관측";
  const classes =
    typeof model.routeClasses === "number" ? `항로 ${numberText(model.routeClasses)}개` : "";
  const anomalies =
    typeof model.anomalyCount === "number" ? `이상 ${numberText(model.anomalyCount)}척` : "";
  routeModelNote.textContent = [earlyWindow, classes, anomalies].filter(Boolean).join(" · ");
}

function setFutureModelSummary(model) {
  if (!model?.available) {
    futureModelNote.textContent = "미래 좌표 모델: 아직 학습 전";
    futureBaselineNote.textContent = "baseline 정보 없음";
    return;
  }
  const horizons = Array.isArray(model.horizons) && model.horizons.length > 0
    ? `${model.horizons.join("/")}h`
    : "-";
  const error =
    typeof model.meanErrorKm === "number" ? `평균 오차 ${model.meanErrorKm.toFixed(2)}km` : "오차 정보 없음";
  futureModelNote.textContent = `미래 좌표 모델: ${horizons} · ${error}`;
  futureBaselineNote.textContent = futureBaselineText(model);
}

function futureBaselineText(model) {
  const errors = model?.errorsKm || {};
  const dead = model?.baselineErrorsKm?.dead_reckoning || {};
  const keys = Object.keys(errors);
  if (keys.length === 0) {
    return "미래 위치 baseline 정보 없음";
  }
  const parts = keys.slice(0, 3).map((key) => {
    const modelError = typeof errors[key] === "number" ? `${errors[key].toFixed(2)}km` : "-";
    const deadError = typeof dead[key] === "number" ? `${dead[key].toFixed(2)}km` : "-";
    return `${key} ${modelError} / DR ${deadError}`;
  });
  return parts.join(" · ");
}

async function refreshMap() {
  const params = new URLSearchParams({
    ship_type: shipTypeSelect.value || "__all__",
    route: routeSelect.value || "__all__",
    anomaly: anomalyOnly.checked ? "1" : "0",
    tracks: trackToggle.checked ? "1" : "0",
    future: futureToggle.checked ? "1" : "0",
    max_ships: "800",
  });
  const response = await fetch(`/api/map-data?${params}`);
  const data = await response.json();
  renderMap(data);
  renderSummary(data);
}

function renderMap(data) {
  [state.routeLayer, state.trackLayer, state.futureLayer, state.shipLayer].forEach((layer) => {
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

  state.futureLayer = L.geoJSON(data.futureTracks, {
    style: (feature) => ({
      color: routeColor(feature.properties.route),
      weight: 2.2,
      opacity: 0.72,
      dashArray: "1 8",
    }),
    onEachFeature: (feature, layer) => {
      layer.bindPopup(futurePopup(feature.properties));
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

  const bounds = combinedBounds([
    state.routeLayer,
    state.trackLayer,
    state.futureLayer,
    state.shipLayer,
  ]);
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
  if (data.routeSummary.length === 0) {
    routeList.innerHTML = `<div class="empty-row">표시할 항로가 없습니다.</div>`;
    return;
  }
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

function renderPerformance(performance) {
  const classMetrics = Array.isArray(performance?.classMetrics) ? performance.classMetrics : [];
  const confusionPairs = Array.isArray(performance?.confusionPairs) ? performance.confusionPairs : [];

  if (classMetrics.length === 0) {
    performanceList.innerHTML = `<div class="empty-row">선종별 성능 데이터가 없습니다.</div>`;
  } else {
    const rows = classMetrics
      .slice()
      .sort((left, right) => (right.support || 0) - (left.support || 0))
      .map(
        (item) => `
          <div class="performance-row">
            <span class="performance-type">${escapeHtml(item.shiptype)}</span>
            <span class="performance-value">${formatPercent(item.f1_score)}</span>
            <span class="performance-value">${formatPercent(item.recall)}</span>
            <span class="performance-value">${numberText(item.support)}</span>
          </div>
        `,
      )
      .join("");
    performanceList.innerHTML = `
      <div class="performance-row header">
        <span>선종</span><span>F1</span><span>Recall</span><span>표본</span>
      </div>
      ${rows}
    `;
  }

  if (confusionPairs.length === 0) {
    confusionList.innerHTML = `<div class="empty-row">혼동 pair 데이터가 없습니다.</div>`;
    return;
  }
  confusionList.innerHTML = confusionPairs
    .slice(0, 8)
    .map(
      (item) => `
        <div class="confusion-row">
          <span class="confusion-pair">${escapeHtml(item.actual)} → ${escapeHtml(item.predicted)}</span>
          <span class="confusion-count">${numberText(item.count)}</span>
        </div>
      `,
    )
    .join("");
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
      <span>예측 선종</span><strong>${escapeHtml(properties.shiptype)}</strong>
      <span>선종 예측 확신도</span><strong>${formatPercent(properties.shiptype_probability)}</strong>
      <span>예측 항로</span><strong>${escapeHtml(properties.route)}</strong>
      <span>항로 예측 확신도</span><strong>${formatPercent(properties.route_probability)}</strong>
      <span>평균 속도</span><strong>${formatValue(properties.mean_sog, "kn")}</strong>
      <span>이상 항로</span><strong>${anomalyText(properties)}</strong>
      <span>이상 점수</span><strong>${formatNumber(properties.anomaly_score, 2)}</strong>
      <span>항로 거리비</span><strong>${formatNumber(properties.route_distance_ratio, 2)}</strong>
      <span>정박지 거리</span><strong>${formatValue(properties.anchorage_distance_km, "km")}</strong>
      <span>크기</span><strong>${shipSizeText(properties)}</strong>
    </div>
  `;
}

function routePopup(properties) {
  return `
    <p class="popup-title">${escapeHtml(properties.route_label)}</p>
    <div class="popup-grid">
      <span>선박 수</span><strong>${numberText(properties.vessel_count)}</strong>
      <span>이상 항로</span><strong>${numberText(properties.anomaly_count)}</strong>
      <span>평균 항로 확신도</span><strong>${formatPercent(properties.avg_route_probability)}</strong>
      <span>평균 선종 확신도</span><strong>${formatPercent(properties.avg_shiptype_probability)}</strong>
      <span>대표 MMSI</span><strong>${escapeHtml(properties.representative_mmsi || "-")}</strong>
      <span>표시 기준</span><strong>${sourceLabel(properties.center_source)}</strong>
    </div>
  `;
}

function futurePopup(properties) {
  const horizons = Array.isArray(properties.horizons) && properties.horizons.length > 0
    ? `${properties.horizons.join(" / ")}시간 후`
    : "-";
  return `
    <p class="popup-title">${escapeHtml(properties.mmsi)}</p>
    <div class="popup-grid">
      <span>예측 선종</span><strong>${escapeHtml(properties.shiptype)}</strong>
      <span>예측 항로</span><strong>${escapeHtml(properties.route)}</strong>
      <span>예측 시간</span><strong>${horizons}</strong>
      <span>기준 시각</span><strong>${escapeHtml(properties.start_timestamp || "-")}</strong>
      <span>모델 평균 오차</span><strong>${formatValue(properties.mean_error_km, "km")}</strong>
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

function methodLabel(method) {
  const labels = {
    mmsi_group_split: "MMSI Group Split",
    nested_mmsi_group_tuning: "MMSI Group Tuning",
  };
  return labels[method] || method || "-";
}

function sourceLabel(source) {
  const labels = {
    representative_actual_track: "실제 대표 항적",
    fallback_cluster_center: "클러스터 중심선",
    selected_ais_tracks: "선택 항적 평균",
  };
  return labels[source] || source || "-";
}

function numberText(value) {
  const number = Number(value || 0);
  return new Intl.NumberFormat("ko-KR").format(number);
}

function formatPercent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
}

function formatNumber(value, digits = 1) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}

function formatValue(value, suffix) {
  return typeof value === "number" ? `${value.toFixed(1)} ${suffix}` : "-";
}

function anomalyText(properties) {
  if (!properties.is_anomaly) {
    return "N";
  }
  const ratio = formatNumber(properties.route_distance_ratio, 2);
  const threshold = formatValue(properties.route_distance_threshold, "km");
  if (ratio !== "-" && threshold !== "-") {
    return `Y (거리비 ${ratio}, 기준 ${threshold})`;
  }
  if (ratio !== "-") {
    return `Y (거리비 ${ratio})`;
  }
  return "Y";
}

function shipSizeText(properties) {
  const length = typeof properties.length === "number" ? `${properties.length.toFixed(0)}m` : "-";
  const width = typeof properties.width === "number" ? `${properties.width.toFixed(0)}m` : "-";
  const draught = typeof properties.draught === "number" ? `${properties.draught.toFixed(1)}m` : "-";
  return `${length} / ${width} / ${draught}`;
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
  routeList.innerHTML = `<div class="empty-row">지도 데이터를 불러오지 못했습니다.</div>`;
});
