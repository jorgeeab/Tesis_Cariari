#!/usr/bin/env python3
"""
Servidor de previsualización de parámetros de excavación del DEM.

Uso:
    python preview_server.py

Luego abrir automaticamente: http://localhost:8765

Requiere haber ejecutado al menos una vez:
    python generar_dem_3d_threejs_satelital.py
(genera output/preview_patch.json)
"""
import json, subprocess, sys, webbrowser, threading, time, math
import socketserver
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

PORT      = 8765
BASE_DIR  = Path(__file__).parent
PATCH_FILE = BASE_DIR / "resultados_refinados" / "preview_patch.json"
SCRIPT     = BASE_DIR / "generar_dem_3d_threejs_satelital.py"
VENDOR_DIR = BASE_DIR / "resultados_refinados" / "_vendor_threejs"
MAP_HTML   = BASE_DIR / "resultados_refinados" / "Refinado_30_ThreeJS_Cortina_Ajustada.html"

# ── Cargar librería Three.js desde vendor local ──────────────────────────────
def _read_vendor(pattern):
    for p in VENDOR_DIR.glob(pattern):
        return p.read_text(encoding="utf-8")
    return ""

THREE_JS  = _read_vendor("three*.min.js")
ORBIT_JS  = _read_vendor("OrbitControls*.js")

# ── Patch JSON ────────────────────────────────────────────────────────────────
def load_patch_json():
    if not PATCH_FILE.exists():
        return None
    return PATCH_FILE.read_text(encoding="utf-8")

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Ajuste de excavación - Cariari</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;
     display:flex;height:100vh;overflow:hidden}

/* ── Sidebar ── */
#sidebar{
  width:270px;min-width:270px;background:#161b22;
  padding:14px 12px;overflow-y:auto;
  display:flex;flex-direction:column;gap:10px;
  border-right:1px solid #30363d;
}
#sidebar h2{color:#58a6ff;font-size:13px;font-weight:600}
.sub{font-size:10px;color:#8b949e;text-transform:uppercase;
     letter-spacing:.5px;margin-top:6px;margin-bottom:2px}
.param{display:flex;flex-direction:column;gap:3px}
.param label{font-size:12px;color:#c9d1d9;display:flex;justify-content:space-between}
.param label strong{color:#79c0ff;min-width:36px;text-align:right}
.param input[type=range]{width:100%;accent-color:#1f6feb;cursor:pointer}
.param-note{font-size:10px;color:#8b949e;line-height:1.35}
#hint{font-size:11px;color:#8b949e;min-height:14px}

/* Botones */
.btn{
  border:none;border-radius:6px;padding:8px;cursor:pointer;
  font-size:12px;width:100%;transition:background .15s;
}
#btn-update{background:#1f6feb;color:#fff}
#btn-update:hover{background:#388bfd}
#btn-run{background:#238636;color:#fff;margin-top:2px}
#btn-run:hover:not(:disabled){background:#2ea043}
#btn-run:disabled{background:#21262d;color:#484f58;cursor:not-allowed}

/* Log / spinner */
#spinner{
  display:none;align-items:center;gap:6px;font-size:11px;color:#58a6ff;
  padding:4px 0;
}
#spinner.on{display:flex}
#spinner-icon{animation:spin 1s linear infinite;display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
#log{
  background:#0d1117;border:1px solid #30363d;border-radius:6px;
  padding:6px 8px;font-size:10px;font-family:monospace;color:#8b949e;
  max-height:220px;overflow-y:auto;display:none;
}
#log.on{display:block}
.ll{color:#c9d1d9;white-space:pre-wrap;word-break:break-all;line-height:1.4}
.ok{color:#3fb950;font-weight:600}
.er{color:#f85149}

/* Canvas */
#wrap{flex:1;position:relative;overflow:hidden}
canvas{display:block!important;width:100%!important;height:100%!important}
#no-data{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:8px;color:#8b949e;
  text-align:center;padding:24px;
}
#no-data code{background:#161b22;padding:4px 10px;border-radius:4px;
               font-size:12px;color:#79c0ff}
</style>
</head>
<body>

<div id="sidebar">
  <h2>Ajuste de excavación Cariari</h2>
  <div id="hint">Cargando...</div>

  <div class="sub">Calles (fondo plano)</div>
  <div class="param">
    <label>Profundidad <strong id="v-sd">1.5 m</strong></label>
    <input type="range" id="sl-sd" min="0.3" max="3.0" step="0.1" value="1.5">
  </div>
  <div class="param">
    <label>Ancho (&times; real) <strong id="v-sw">1.0&times;</strong></label>
    <input type="range" id="sl-sw" min="0.4" max="3.0" step="0.1" value="1.0">
  </div>
  <div class="param">
    <label>Suavidad del borde <strong id="v-sr">7.0 m</strong></label>
    <input type="range" id="sl-sr" min="1" max="20" step="0.5" value="7.0">
  </div>
  <div class="param" style="flex-direction:row;align-items:center;justify-content:space-between">
    <label>Color de calle</label>
    <input type="color" id="sl-scol" value="#2b2a28" style="width:42px;height:24px;border:none;background:none;cursor:pointer">
  </div>

  <div class="sub">Ríos</div>
  <div class="param">
    <label>Profundidad <strong id="v-rd">2.5 m</strong></label>
    <input type="range" id="sl-rd" min="0.5" max="6.0" step="0.1" value="2.5">
  </div>
  <div class="param">
    <label>Ancho del cauce <strong id="v-rw">4.0 m</strong></label>
    <input type="range" id="sl-rw" min="1" max="15" step="0.5" value="4.0">
  </div>
  <div class="param" style="flex-direction:row;align-items:center;justify-content:space-between">
    <label>Color de río</label>
    <input type="color" id="sl-rcol" value="#2f6fb0" style="width:42px;height:24px;border:none;background:none;cursor:pointer">
  </div>

  <div class="sub">Infiltracion</div>
  <div class="param">
    <label>Suelo abierto <strong id="v-inf-base">22%</strong></label>
    <input type="range" id="sl-inf-base" min="0" max="100" step="1" value="22">
  </div>
  <div class="param">
    <label>Vegetacion <strong id="v-inf-veg">90%</strong></label>
    <input type="range" id="sl-inf-veg" min="0" max="100" step="1" value="90">
  </div>
  <div class="param">
    <label>Edificios <strong id="v-inf-bld">2%</strong></label>
    <input type="range" id="sl-inf-bld" min="0" max="100" step="1" value="2">
  </div>
  <div class="param">
    <label>Calles <strong id="v-inf-street">6%</strong></label>
    <input type="range" id="sl-inf-street" min="0" max="100" step="1" value="6">
  </div>
  <div class="param-note">La escorrentia se calcula automaticamente como 100 menos la infiltracion.</div>

  <div class="sub">Visual</div>
  <div class="param">
    <label>Vista base</label>
    <select id="sl-pal" style="width:100%;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px;font-size:12px;cursor:pointer">
      <option value="sat" selected>Imagen satelital</option>
      <option value="elev">Elevación (verde-tan)</option>
      <option value="heat">Mapa de calor (jet)</option>
      <option value="terrain">Terreno (agua-verde-nieve)</option>
      <option value="gray">Escala de grises</option>
    </select>
  </div>
  <div class="param">
    <label>Contraste color <strong id="v-ct">1.0</strong></label>
    <input type="range" id="sl-ct" min="0.4" max="3.0" step="0.1" value="1.0">
  </div>
  <div class="param">
    <label>Exageración vertical <strong id="v-ex">1.0×</strong></label>
    <input type="range" id="sl-ex" min="0.5" max="6.0" step="0.1" value="1.0">
  </div>
  <div class="param">
    <label>Pixelaje (resolución) <strong id="v-rs">1×</strong></label>
    <input type="range" id="sl-rs" min="1" max="5" step="1" value="1">
  </div>
  <div class="param" style="flex-direction:row;align-items:center;justify-content:space-between">
    <label><input type="checkbox" id="sl-bld" checked style="vertical-align:middle"> Casas (edificios)</label>
    <input type="color" id="sl-bcol" value="#d9c2a6" style="width:42px;height:24px;border:none;background:none;cursor:pointer">
  </div>

  <button class="btn" id="btn-update">&#8635; Actualizar vista (o suelta slider)</button>

  <div class="sub" style="margin-top:4px">Mapa principal</div>
  <button class="btn" id="btn-run">&#9654; Aplicar al mapa principal</button>
  <div id="spinner"><span id="spinner-icon">&#9696;</span> Aplicando...</div>
  <div id="log"></div>
</div>

<div id="wrap">
  <div id="no-data" style="display:none">
    <div>&#9888;&#65039; Sin datos de previsualización.</div>
    <div>Ejecuta primero el script principal:</div>
    <code>python generar_dem_3d_threejs_satelital.py</code>
  </div>
</div>

<script>
// ── Embedded vendor + patch ───────────────────────────────────────────────
{THREE_JS}
</script>
<script>
{ORBIT_JS}
</script>
<script>
const _PATCH = {PATCH_JSON};
</script>
<script>
// ── Gaussian blur (1-D separable) ─────────────────────────────────────────
function gaussBlur(data, nx, ny, sigma){
  if(sigma<=0) return data;
  const r=Math.ceil(sigma*2.5);
  const k=[];let ks=0;
  for(let i=-r;i<=r;i++){const v=Math.exp(-.5*i*i/(sigma*sigma));k.push(v);ks+=v;}
  for(let i=0;i<k.length;i++) k[i]/=ks;
  const tmp=new Float64Array(data.length);
  const out=new Float64Array(data.length);
  // horizontal
  for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){
    let s=0;
    for(let d=-r;d<=r;d++) s+=data[j*nx+Math.max(0,Math.min(nx-1,i+d))]*k[d+r];
    tmp[j*nx+i]=s;
  }
  // vertical
  for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){
    let s=0;
    for(let d=-r;d<=r;d++) s+=tmp[Math.max(0,Math.min(ny-1,j+d))*nx+i]*k[d+r];
    out[j*nx+i]=s;
  }
  return out;
}

// ── Compute heights from patch + sliders ──────────────────────────────────
// sd = prof. calle (m), sw = ancho calle (×real), sr = suavidad borde (m)
// rd = prof. río (m),   rw = ancho cauce (m, sigma gaussiano)
function computeHeights(p, sd, sw, sr, rd, rw){
  const ny=p.ny, nx=p.nx, cm=p.cell_m;
  const street=new Float64Array(ny*nx);

  for(let idx=0;idx<ny*nx;idx++){
    // Canal de calle con FONDO PLANO:
    //   dentro del ancho efectivo (semiancho base × sw) → profundidad constante sd
    //   en el borde (sr metros) → rampa lineal sd → 0
    const dist   = p.street_dist[idx];          // distancia al eje (m)
    const halfEff= p.street_half[idx] * sw;      // semiancho efectivo (m)
    let off;
    if(dist <= halfEff)            off = sd;                         // fondo plano
    else if(dist < halfEff + sr)   off = sd * (1 - (dist-halfEff)/sr); // pared
    else                           off = 0;
    street[idx]=Math.max(0,off);
  }

  // Suavizado leve del borde para eliminar escalera de la rasterización
  const sigma_px=Math.max(0.6, sr/cm*0.18);
  const bstreet=gaussBlur(street,nx,ny,sigma_px);

  const sigma_m = rw;   // ancho del cauce = sigma del perfil gaussiano (igual que Python)
  const h     =new Float32Array(ny*nx);
  const sMask =new Float32Array(ny*nx);   // máscara calle 0-1 (para color)
  const rMask =new Float32Array(ny*nx);   // máscara río  0-1 (para color)
  for(let idx=0;idx<ny*nx;idx++){
    const rv=p.river_dist[idx];
    const river_drop = Math.max(0, rd*Math.exp(-.5*(rv/sigma_m)**2));
    const street_drop= Math.max(0, bstreet[idx]);
    sMask[idx]= sd>0 ? Math.min(1, street_drop/sd) : 0;
    rMask[idx]= rd>0 ? Math.min(1, river_drop /rd) : 0;
    h[idx] = p.dem[idx] - river_drop - street_drop;
  }
  return {h, sMask, rMask};
}

// ── Bilinear upsample de una grilla nx×ny por factor S ──────────────────────
function upsampleBilinear(src, nx, ny, S){
  if(S<=1) return {data:src, NX:nx, NY:ny};
  const NX=(nx-1)*S+1, NY=(ny-1)*S+1;
  const out=new Float32Array(NX*NY);
  for(let J=0;J<NY;J++){
    const fy=J/S, j0=Math.floor(fy), j1=Math.min(ny-1,j0+1), ty=fy-j0;
    for(let I=0;I<NX;I++){
      const fx=I/S, i0=Math.floor(fx), i1=Math.min(nx-1,i0+1), tx=fx-i0;
      const a=src[j0*nx+i0], b=src[j0*nx+i1], c=src[j1*nx+i0], d=src[j1*nx+i1];
      out[J*NX+I]=(a*(1-tx)+b*tx)*(1-ty)+(c*(1-tx)+d*tx)*ty;
    }
  }
  return {data:out, NX, NY};
}

// ── Paletas de color ────────────────────────────────────────────────────────
function paletteColor(name, t, contrast){
  // t en [0,1]; contrast aplica una curva (gamma) para realzar
  t=Math.max(0,Math.min(1,t));
  if(contrast!==1){ t = Math.pow(t, 1/contrast); }
  let r,g,b;
  if(name==='heat'){          // mapa de calor estilo jet: azul→cian→verde→amarillo→rojo
    if(t<.25){const f=t/.25; r=0; g=f; b=1;}
    else if(t<.5){const f=(t-.25)/.25; r=0; g=1; b=1-f;}
    else if(t<.75){const f=(t-.5)/.25; r=f; g=1; b=0;}
    else{const f=(t-.75)/.25; r=1; g=1-f; b=0;}
  } else if(name==='gray'){   // escala de grises
    r=g=b=.15+.8*t;
  } else if(name==='terrain'){// satelital-ish: azul agua → verde → marrón → blanco
    if(t<.15){const f=t/.15; r=.1+.1*f; g=.3+.3*f; b=.6+.2*f;}
    else if(t<.5){const f=(t-.15)/.35; r=.2+.25*f; g=.6-.05*f; b=.25-.05*f;}
    else if(t<.8){const f=(t-.5)/.3; r=.45+.3*f; g=.5-.1*f; b=.2+.05*f;}
    else{const f=(t-.8)/.2; r=.75+.25*f; g=.4+.6*f; b=.25+.75*f;}
  } else {                    // 'elev' (defecto): verde→tan→gris
    if(t<.35){const f=t/.35; r=.15+.45*f; g=.42+.08*f; b=.10+.05*f;}
    else if(t<.7){const f=(t-.35)/.35; r=.60+.15*f; g=.50-.05*f; b=.15+.10*f;}
    else{const f=(t-.7)/.3; r=.75+.10*f; g=.45-.05*f; b=.25+.15*f;}
  }
  return [r,g,b];
}

// ── Three.js scene ────────────────────────────────────────────────────────
const wrap=document.getElementById('wrap');
const noData=document.getElementById('no-data');
const hintEl=document.getElementById('hint');
let scene,camera,renderer,controls,mesh;

function initThree(){
  scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0d1117);
  scene.fog=new THREE.FogExp2(0x0d1117,.0012);

  const W=wrap.clientWidth, H=wrap.clientHeight;
  camera=new THREE.PerspectiveCamera(50,W/H,.5,5000);
  camera.position.set(0,160,260);
  camera.lookAt(0,0,0);

  renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.setSize(W,H);
  wrap.appendChild(renderer.domElement);

  controls=new THREE.OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.dampingFactor=.1;
  controls.autoRotate=true;controls.autoRotateSpeed=.5;
  controls.addEventListener('start',()=>controls.autoRotate=false);

  const amb=new THREE.AmbientLight(0xffffff,.55);scene.add(amb);
  const sun=new THREE.DirectionalLight(0xfff5e0,1.1);
  sun.position.set(100,280,80);scene.add(sun);
  const fill=new THREE.DirectionalLight(0xc0d8ff,.3);
  fill.position.set(-100,80,-60);scene.add(fill);

  window.addEventListener('resize',()=>{
    const W2=wrap.clientWidth,H2=wrap.clientHeight;
    camera.aspect=W2/H2;camera.updateProjectionMatrix();
    renderer.setSize(W2,H2);
  });
  (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();
}

// ── Textura satelital: cargada una vez a un canvas para muestreo por vértice ─
let SAT=null;   // {data, w, h}
function loadSatellite(b64, cb){
  if(!b64){ cb(); return; }
  const im=new Image();
  im.onload=()=>{
    const cv=document.createElement('canvas');
    cv.width=im.naturalWidth; cv.height=im.naturalHeight;
    const cx=cv.getContext('2d');
    cx.drawImage(im,0,0);
    try{
      const d=cx.getImageData(0,0,cv.width,cv.height);
      SAT={data:d.data, w:cv.width, h:cv.height};
    }catch(e){ SAT=null; }
    cb();
  };
  im.onerror=()=>{ SAT=null; cb(); };
  im.src='data:image/png;base64,'+b64;
}
function sampleSat(u,v){
  // u,v en [0,1]; fila 0 = lats_crop[0] = col 0 = lons_crop[0] (ya orientado en Python)
  const px=Math.min(SAT.w-1, Math.max(0, Math.floor(u*SAT.w)));
  const py=Math.min(SAT.h-1, Math.max(0, Math.floor(v*SAT.h)));
  const k=(py*SAT.w+px)*4;
  return [SAT.data[k]/255, SAT.data[k+1]/255, SAT.data[k+2]/255];
}

// hex '#rrggbb' → [r,g,b] 0-1
function hex2rgb(h){
  const n=parseInt(h.replace('#',''),16);
  return [((n>>16)&255)/255, ((n>>8)&255)/255, (n&255)/255];
}

let buildingsGroup=null;

// ── Build terrain mesh (explicit BufferGeometry) ───────────────────────────
// opts = {exag, palette, res, contrast, streetCol, riverCol, showBld, bldCol}
function buildMesh(p, heights, sMask, rMask, opts){
  if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();mesh=null;}
  const W=p.widthM, H=Math.abs(p.heightM);
  const exag=opts.exag, palette=opts.palette, S=opts.res, contrast=opts.contrast;
  const sCol=hex2rgb(opts.streetCol), rCol=hex2rgb(opts.riverCol);
  const useSat = (palette==='sat' && SAT);

  // Upsample (pixelaje)
  const upH=upsampleBilinear(heights, p.nx, p.ny, S);
  const upS=upsampleBilinear(sMask,   p.nx, p.ny, S);
  const upR=upsampleBilinear(rMask,   p.nx, p.ny, S);
  const Hg=upH.data, Sg=upS.data, Rg=upR.data, nx=upH.NX, ny=upH.NY;

  let hMin=Infinity, hMax=-Infinity;
  for(let k=0;k<Hg.length;k++){
    const v=Hg[k]; if(isFinite(v)){ if(v<hMin)hMin=v; if(v>hMax)hMax=v; }
  }
  if(!isFinite(hMin)){ hMin=0; hMax=1; }
  const hRng=Math.max(1, hMax-hMin);
  const hMid=(hMin+hMax)/2;
  buildMesh._hMid=hMid; buildMesh._exag=exag;

  const verts=new Float32Array(nx*ny*3);
  const cols =new Float32Array(nx*ny*3);
  for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){
    const idx=j*nx+i;
    const h=Hg[idx];
    const u=i/(nx-1), w=j/(ny-1);
    verts[idx*3]   = -W/2 + u*W;
    verts[idx*3+1] = (h - hMid)*exag;
    verts[idx*3+2] = -H/2 + w*H;

    // Color base: satelital o paleta de elevación
    let r,g,b;
    if(useSat){ const c=sampleSat(u,w); r=c[0]; g=c[1]; b=c[2]; }
    else      { const c=paletteColor(palette,(h-hMin)/hRng,contrast); r=c[0]; g=c[1]; b=c[2]; }

    // Mezclar color de río y de calle según máscaras
    const rm=Math.min(1, Rg[idx]);
    r=r*(1-rm)+rCol[0]*rm; g=g*(1-rm)+rCol[1]*rm; b=b*(1-rm)+rCol[2]*rm;
    const sm=Math.min(1, Sg[idx]);
    r=r*(1-sm)+sCol[0]*sm; g=g*(1-sm)+sCol[1]*sm; b=b*(1-sm)+sCol[2]*sm;

    cols[idx*3]=r; cols[idx*3+1]=g; cols[idx*3+2]=b;
  }

  const indices=[];
  for(let j=0;j<ny-1;j++) for(let i=0;i<nx-1;i++){
    const a=j*nx+i, b=j*nx+i+1, c=(j+1)*nx+i, d=(j+1)*nx+i+1;
    indices.push(a,c,b, b,c,d);
  }

  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts,3));
  geo.setAttribute('color',    new THREE.BufferAttribute(cols,3));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  const mat=new THREE.MeshLambertMaterial({vertexColors:true, side:THREE.DoubleSide});
  mesh=new THREE.Mesh(geo,mat);
  scene.add(mesh);

  // Edificios (casas)
  buildBuildings(p, heights, opts, hMid, exag);

  if(!buildMesh._fitted){
    buildMesh._fitted=true;
    const span=Math.max(W,H);
    camera.near=span/1000; camera.far=span*10; camera.updateProjectionMatrix();
    camera.position.set(span*0.55, span*0.45, span*0.55);
    controls.target.set(0,0,0);
    controls.update();
  }
  buildMesh._lastInfo=`elev ${hMin.toFixed(0)}-${hMax.toFixed(0)}m · malla ${nx}x${ny}`;
}

// ── Edificios extruidos sobre el terreno ────────────────────────────────────
function sampleHeightAt(heights, nx, ny, W, H, x, z){
  // x,z en metros (centro=0) → índice de grilla → altura base
  const fi=(x + W/2)/W*(nx-1), fj=(z + H/2)/H*(ny-1);
  const i=Math.min(nx-1, Math.max(0, Math.round(fi)));
  const j=Math.min(ny-1, Math.max(0, Math.round(fj)));
  return heights[j*nx+i];
}
function buildBuildings(p, heights, opts, hMid, exag){
  if(buildingsGroup){ scene.remove(buildingsGroup);
    buildingsGroup.traverse(o=>{ if(o.geometry)o.geometry.dispose(); if(o.material)o.material.dispose(); });
    buildingsGroup=null; }
  if(!opts.showBld || !p.buildings || !p.buildings.length) return;
  const W=p.widthM, H=Math.abs(p.heightM);
  buildingsGroup=new THREE.Group();
  const bcol=new THREE.Color(opts.bldCol);
  const mat=new THREE.MeshLambertMaterial({color:bcol, side:THREE.DoubleSide});
  for(const bld of p.buildings){
    const pts=bld.pts;
    if(!pts || pts.length<3) continue;
    // Centroide para muestrear altura del terreno
    let cx=0,cz=0; for(const q of pts){cx+=q[0];cz+=q[1];} cx/=pts.length; cz/=pts.length;
    const hTer=sampleHeightAt(heights,p.nx,p.ny,W,H,cx,cz);
    const yBase=(hTer-hMid)*exag;
    const hh=Math.max(2, bld.h)*exag;
    // Shape en (x, -z): tras rotateX(-90°) → world (x, altura, z) alineado al terreno
    const shape=new THREE.Shape();
    shape.moveTo(pts[0][0], -pts[0][1]);
    for(let k=1;k<pts.length;k++) shape.lineTo(pts[k][0], -pts[k][1]);
    shape.closePath();
    let geo;
    try{ geo=new THREE.ExtrudeGeometry(shape,{depth:hh,bevelEnabled:false,steps:1}); }
    catch(e){ continue; }
    // Shape está en plano XY; extrusión en +Z. Rotar para que altura quede en +Y.
    geo.rotateX(-Math.PI/2);
    geo.translate(0, yBase, 0);
    buildingsGroup.add(new THREE.Mesh(geo, mat));
  }
  scene.add(buildingsGroup);
}

// ── UI ────────────────────────────────────────────────────────────────────
const sliders=['sl-sd','sl-sw','sl-sr','sl-rd','sl-rw','sl-ct','sl-ex','sl-rs'];
const infiltrationSliders=['sl-inf-base','sl-inf-veg','sl-inf-bld','sl-inf-street'];

function getVals(){
  return {
    sd:parseFloat(document.getElementById('sl-sd').value),
    sw:parseFloat(document.getElementById('sl-sw').value),
    sr:parseFloat(document.getElementById('sl-sr').value),
    rd:parseFloat(document.getElementById('sl-rd').value),
    rw:parseFloat(document.getElementById('sl-rw').value),
    ct:parseFloat(document.getElementById('sl-ct').value),
    ex:parseFloat(document.getElementById('sl-ex').value),
    rs:parseInt(document.getElementById('sl-rs').value),
    pal:document.getElementById('sl-pal').value,
    scol:document.getElementById('sl-scol').value,
    rcol:document.getElementById('sl-rcol').value,
    bcol:document.getElementById('sl-bcol').value,
    bld:document.getElementById('sl-bld').checked,
    infBase:parseFloat(document.getElementById('sl-inf-base').value),
    infVeg:parseFloat(document.getElementById('sl-inf-veg').value),
    infBld:parseFloat(document.getElementById('sl-inf-bld').value),
    infStreet:parseFloat(document.getElementById('sl-inf-street').value),
  };
}
function setControlValue(id,value){
  const el=document.getElementById(id);
  if(!el || value===undefined || value===null) return;
  el.value=String(value);
}
function applyPatchSettings(settings){
  if(!settings) return;
  setControlValue('sl-sd',settings.street_depth);
  setControlValue('sl-sw',settings.street_width);
  setControlValue('sl-sr',settings.street_ramp);
  setControlValue('sl-rd',settings.river_depth);
  setControlValue('sl-rw',settings.river_width);
  const inf=settings.infiltration_pct || {};
  const ro=settings.runoff_pct || {};
  setControlValue('sl-inf-base',inf.base ?? (ro.base===undefined ? undefined : 100-ro.base));
  setControlValue('sl-inf-veg',inf.vegetation ?? (ro.vegetation===undefined ? undefined : 100-ro.vegetation));
  setControlValue('sl-inf-bld',inf.building ?? (ro.building===undefined ? undefined : 100-ro.building));
  setControlValue('sl-inf-street',inf.street ?? (ro.street===undefined ? undefined : 100-ro.street));
}
function syncLabels(){
  const v=getVals();
  document.getElementById('v-sd').textContent=v.sd.toFixed(1)+' m';
  document.getElementById('v-sw').textContent=v.sw.toFixed(1)+'×';
  document.getElementById('v-sr').textContent=v.sr.toFixed(1)+' m';
  document.getElementById('v-rd').textContent=v.rd.toFixed(1)+' m';
  document.getElementById('v-rw').textContent=v.rw.toFixed(1)+' m';
  document.getElementById('v-ct').textContent=v.ct.toFixed(1);
  document.getElementById('v-ex').textContent=v.ex.toFixed(1)+'×';
  document.getElementById('v-rs').textContent=v.rs+'×';
  document.getElementById('v-inf-base').textContent=v.infBase.toFixed(0)+'%';
  document.getElementById('v-inf-veg').textContent=v.infVeg.toFixed(0)+'%';
  document.getElementById('v-inf-bld').textContent=v.infBld.toFixed(0)+'%';
  document.getElementById('v-inf-street').textContent=v.infStreet.toFixed(0)+'%';
}
function redraw(){
  if(!window._patch) return;
  const t0=performance.now();
  const v=getVals();
  const r=computeHeights(window._patch,v.sd,v.sw,v.sr,v.rd,v.rw);
  buildMesh(window._patch, r.h, r.sMask, r.rMask,
            {exag:v.ex, palette:v.pal, res:v.rs, contrast:v.ct,
             streetCol:v.scol, riverCol:v.rcol, showBld:v.bld, bldCol:v.bcol});
  hintEl.textContent=`${(performance.now()-t0).toFixed(0)}ms · ${buildMesh._lastInfo}`;
}

sliders.forEach(id=>{
  const el=document.getElementById(id);
  el.addEventListener('input',syncLabels);
  el.addEventListener('change',redraw);  // auto-redraw on release
});
infiltrationSliders.forEach(id=>{
  document.getElementById(id).addEventListener('input',syncLabels);
});
['sl-pal','sl-scol','sl-rcol','sl-bcol','sl-bld'].forEach(id=>{
  document.getElementById(id).addEventListener('change',redraw);
});
document.getElementById('sl-scol').addEventListener('input',redraw);
document.getElementById('sl-rcol').addEventListener('input',redraw);
document.getElementById('sl-bcol').addEventListener('input',redraw);
document.getElementById('btn-update').addEventListener('click',redraw);

// ── Aplicar al mapa principal ───────────────────────────────────────────────
const btnRun=document.getElementById('btn-run');
const spinner=document.getElementById('spinner');
const logEl=document.getElementById('log');

function appendLog(txt,cls){
  const d=document.createElement('div');
  d.className='ll'+(cls?' '+cls:'');
  d.textContent=txt;
  logEl.appendChild(d);
  logEl.scrollTop=logEl.scrollHeight;
}

btnRun.addEventListener('click',()=>{
  const v=getVals();
  const params=new URLSearchParams({
    street_depth:v.sd.toFixed(1),
    street_width:v.sw.toFixed(1),
    street_ramp:v.sr.toFixed(1),
    river_depth:v.rd.toFixed(1),
    river_width:v.rw.toFixed(1),
    infiltration_base_pct:v.infBase.toFixed(0),
    infiltration_vegetation_pct:v.infVeg.toFixed(0),
    infiltration_building_pct:v.infBld.toFixed(0),
    infiltration_street_pct:v.infStreet.toFixed(0),
  });
  const url='/run?'+params.toString();

  btnRun.disabled=true;
  spinner.classList.add('on');
  logEl.innerHTML='';
  logEl.classList.add('on');
  appendLog(`> python generar_dem_3d_threejs_satelital.py --street-depth ${v.sd.toFixed(1)} --street-width ${v.sw.toFixed(1)} --street-ramp ${v.sr.toFixed(1)} --river-depth ${v.rd.toFixed(1)} --river-width ${v.rw.toFixed(1)} --infiltration-base-pct ${v.infBase.toFixed(0)} --infiltration-vegetation-pct ${v.infVeg.toFixed(0)} --infiltration-building-pct ${v.infBld.toFixed(0)} --infiltration-street-pct ${v.infStreet.toFixed(0)}`);

  const es=new EventSource(url);
  es.onmessage=e=>{
    if(e.data==='__done__'){
      es.close();
      spinner.classList.remove('on');
      btnRun.disabled=false;
      appendLog('Mapa principal actualizado.','ok');
      // Link que sirve el mapa DESDE este server (evita el :8000)
      const a=document.createElement('a');
      a.href='/mapa?t='+Date.now();
      a.target='_blank';
      a.textContent='Abrir mapa principal actualizado';
      a.style.cssText='display:block;margin-top:6px;padding:7px;background:#8957e5;color:#fff;text-align:center;border-radius:6px;text-decoration:none;font-size:12px';
      logEl.appendChild(a);
      logEl.scrollTop=logEl.scrollHeight;
      return;
    }
    if(e.data.startsWith('ERROR:')){appendLog(e.data,'er');return;}
    appendLog(e.data);
  };
  es.onerror=()=>{
    es.close();
    spinner.classList.remove('on');
    btnRun.disabled=false;
    appendLog('❌ Error de conexión. El servidor puede haber terminado.','er');
  };
});

// ── Init ──────────────────────────────────────────────────────────────────
initThree();
const p=_PATCH;
if(!p){
  syncLabels();
  noData.style.display='flex';
  hintEl.textContent='Sin datos — ejecuta el script principal primero.';
} else {
  window._patch=p;
  applyPatchSettings(p.settings);
  syncLabels();
  hintEl.textContent=`Parche ${p.nx}×${p.ny}px · ${p.widthM}×${Math.abs(p.heightM)} m`;
  // Cargar satelital primero; cuando esté listo (o falle), renderizar
  loadSatellite(p.sat_b64, ()=>{
    if(!SAT){
      // sin satelital → cambiar la vista base por defecto a Elevación
      const sel=document.getElementById('sl-pal');
      if(sel && sel.value==='sat') sel.value='elev';
    }
    redraw();
  });
}
</script>
</body>
</html>
"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    _patch_json = None   # class-level cache

    def log_message(self, fmt, *a):
        pass  # silence request log

    def _send(self, code, ctype, body_bytes):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body_bytes)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/':
            patch_json = Handler._patch_json or load_patch_json() or 'null'
            Handler._patch_json = patch_json
            page = HTML.replace('{THREE_JS}', THREE_JS) \
                        .replace('{ORBIT_JS}', ORBIT_JS) \
                        .replace('{PATCH_JSON}', patch_json)
            self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'))

        elif parsed.path == '/mapa':
            # Sirve el HTML 3D generado desde ESTE server (evita depender de :8000)
            if not MAP_HTML.exists():
                self._send(404, 'text/plain; charset=utf-8',
                           'Mapa no generado aún. Pulsa "Aplicar al mapa principal".'.encode('utf-8'))
            else:
                data = MAP_HTML.read_bytes()
                self._send(200, 'text/html; charset=utf-8', data)

        elif parsed.path == '/run':
            qs = parse_qs(parsed.query)
            def getf(key, default):
                return float(qs.get(key, [default])[0])
            sd = getf('street_depth', 1.5)
            sw = getf('street_width', 1.0)
            sr = getf('street_ramp',  7.0)
            rd = getf('river_depth',  2.5)
            rw = getf('river_width',  4.0)
            inf_base = getf('infiltration_base_pct', 100.0 - getf('runoff_base_pct', 78.0))
            inf_veg = getf('infiltration_vegetation_pct', 100.0 - getf('runoff_vegetation_pct', 10.0))
            inf_bld = getf('infiltration_building_pct', 100.0 - getf('runoff_building_pct', 98.0))
            inf_street = getf('infiltration_street_pct', 100.0 - getf('runoff_street_pct', 94.0))

            cmd = [sys.executable, str(SCRIPT),
                   '--street-depth', f'{sd:.1f}',
                   '--street-width', f'{sw:.1f}',
                   '--street-ramp',  f'{sr:.1f}',
                   '--river-depth',  f'{rd:.1f}',
                   '--river-width',  f'{rw:.1f}',
                   '--infiltration-base-pct', f'{inf_base:.0f}',
                   '--infiltration-vegetation-pct', f'{inf_veg:.0f}',
                   '--infiltration-building-pct', f'{inf_bld:.0f}',
                   '--infiltration-street-pct', f'{inf_street:.0f}']

            self.send_response(200)
            self.send_header('Content-Type',  'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                import os as _os
                env = {**_os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8', errors='replace',
                                        env=env, cwd=str(BASE_DIR))
                for line in proc.stdout:
                    msg = json.dumps(line.rstrip())
                    self.wfile.write(f'data: {msg}\n\n'.encode())
                    self.wfile.flush()
                proc.wait()
                # Reload patch after run
                Handler._patch_json = load_patch_json()
            except Exception as exc:
                err = json.dumps(f'ERROR: {exc}')
                self.wfile.write(f'data: {err}\n\n'.encode())
                self.wfile.flush()

            self.wfile.write(b'data: __done__\n\n')
            self.wfile.flush()

        else:
            self._send(404, 'text/plain', b'Not found')


class ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import io, sys as _sys
    # Force UTF-8 stdout so emojis don't crash on cp1252 terminals
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    # Pre-cache patch
    Handler._patch_json = load_patch_json()
    if not Handler._patch_json:
        print("[!] preview_patch.json no encontrado.")
        print("    Ejecuta primero: python generar_dem_3d_threejs_satelital.py")
        print("    El servidor igual iniciara — el preview estara vacio.\n")
    else:
        print(f"[OK] Parche cargado: {PATCH_FILE}")

    if not THREE_JS:
        print(f"[!] No se encontro three.js en {VENDOR_DIR}")

    server = ThreadedServer(('', PORT), Handler)
    url = f'http://localhost:{PORT}'
    print(f"\n>>> Preview server corriendo en {url}")
    print("    Ctrl+C para detener.\n")

    # Abrir browser con pequeño delay para que el servidor esté listo
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
