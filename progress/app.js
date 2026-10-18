/* Local artifact browser. No network APIs or external libraries. */
(() => {
  'use strict';
  const D=window.RESEARCH_DATA, test=D.tests.find(t=>t.id===document.body.dataset.test);
  const refN=D.ot.referenceN, checkN=D.ot.checkReferenceN, movieRefN=D.ot.movieReferenceN;
  let transport=null, movieSelection='';
  const roles=['glm','ct','truth'], labels={glm:'GLM',ct:'CT',truth:'The Truth'};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const links=items=>`<div class="link-list">${items.map(([label,url])=>`<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`).join('')}</div>`;
  const stats=items=>`<dl class="stats">${items.map(([label,value])=>`<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`;
  const checks=items=>`<ul class="checks">${items.map(t=>`<li>${esc(t)}</li>`).join('')}</ul>`;
  const img=(src,alt,extra='')=>`<a class="image-link ${extra}" href="${esc(src)}" data-zoom data-caption="${esc(alt)}"><img class="media-image" loading="lazy" src="${esc(src)}" alt="${esc(alt)}"></a>`;
  const card=(title,body,cls='')=>`<section class="card ${cls}"><h3>${esc(title)}</h3>${body}</section>`;
  const note=role=>`<section class="card notes-card"><label for="note-${role}">Working notes</label><textarea id="note-${role}" data-note="${role}" placeholder="Questions, observations, next ideas…"></textarea><span class="note-state" id="note-state-${role}">Saved in this browser only · separate from measured results</span></section>`;
  function column(role,subtitle,status,body){return `<section class="column ${role}" aria-label="${labels[role]}"><div class="column-head"><div class="column-title"><h2>${labels[role]}</h2><span class="status-pill">${esc(status)}</span></div><p id="subtitle-${role}">${esc(subtitle)}</p></div>${body}${note(role)}</section>`;}
  document.querySelector('#intro').innerHTML=`<div class="eyebrow">EXPERIMENT ${test.number} <span class="badge">${esc(test.status)}</span></div><div class="page-heading"><div><h1>${esc(test.title)}</h1><p>${esc(test.description)}</p></div><a class="method-link" href="${esc(test.document)}" target="_blank" rel="noopener">Methods & records <span>↗</span></a></div><div class="chips">${test.chips.map(c=>`<span class="chip">${esc(c)}</span>`).join('')}</div>`;
  document.querySelector('#build-stamp').textContent=`Built from saved artifacts · ${new Date(D.built).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'})}`;

  let selection={n:64,field:'density',time:3};
  try{const p=new URLSearchParams(location.search);if([64,128,256,512].includes(+p.get('n')))selection.n=+p.get('n');if(p.get('field')==='magnetic')selection.field='magnetic';if(p.has('time')&&[0,1,2,3].includes(+p.get('time')))selection.time=+p.get('time');}catch{}
  function snapshot(role){return `<section class="card snapshot-card"><div class="snapshot-heading"><span data-field-title>Density</span><span data-time-title>t = 0.500</span></div><a class="image-link" id="snapshot-link-${role}" data-zoom href="#"><img class="snapshot-image" id="snapshot-${role}" alt=""></a><div class="snapshot-foot"><div class="scale" data-scale></div><div class="scale-labels"><span data-scale-low></span><span>Shared scale · x, y ∈ [0, 1]</span><span data-scale-high></span></div><p class="snapshot-caption" id="snapshot-caption-${role}"></p></div></section>`;}
  function otColumn(role){
    let content=snapshot(role);
    content+=card('The run, in motion',`<video id="movie-${role}" controls playsinline preload="metadata" aria-label="${labels[role]} movie"></video><div class="media-caption" id="movie-caption-${role}"></div><button class="quiet-button" data-play-all>▶ Play all three</button><span class="movie-status" aria-live="polite">Play on any player starts all three together</span>`,'movie-card');
    if(role!=='truth'){
      content+=card('Difference from the reference',`<div id="norms-${role}"></div><p class="small">Relative norms (%) against Athena++ ${refN}² on the selected grid. Magnetic errors use the full vector.</p>`);
    }else{
      content+=card('A reference, not an exact answer',`<div id="truth-statistics"></div><p id="reference-warning" class="notice"></p><div id="reference-sensitivity"></div><p class="small">Independent Athena++ CT · HLLD · PPM / RK3 · double-precision evolution. VTK snapshots store Float32.</p>`);
    }
    content+=card('Which scales survive?',`<a class="image-link" data-zoom id="spectrum-link-${role}" href="#"><img id="spectrum-${role}" class="media-image" loading="lazy" alt="${labels[role]} kinetic and magnetic spectra at t=0.5"></a><p class="small">Final state, t = 0.5 · complete Fourier shells below Nyquist. Power alone does not measure phase accuracy.</p>`);
    content+=card('Where the energy goes',`<a class="image-link" data-zoom id="energy-link-${role}" href="#"><img id="energy-${role}" class="media-image" loading="lazy" alt="${labels[role]} energy history"></a><p class="small">${role==='truth'?'Restricted thermal energy includes unresolved kinetic and magnetic energy. Four reference checkpoints are shown.':'Energy exchanges include ideal MHD dynamics. These curves alone do not isolate numerical resistivity.'}</p>`);
    if(role!=='truth'){
      content+=card('Run record',`<div id="run-record-${role}"></div><p class="small">One exploratory timing. Step time excludes compilation, CFL checks and output; some analysis overlapped the 512² run. A repeated timing runner is prepared but has not been executed.</p><div id="run-links-${role}"></div>${links([['CPU benchmark audit','../scripts/CPU_BENCHMARKS.md']])}`);
      content+=card('Checks completed',checks(role==='glm'?['Finite, positive saved states','Conservative mass and energy updates','Fourier power and real-space norms agree']:['Native face fields saved at every output','Face divergence checked independently','Fourier power and real-space norms agree'])+`<p class="small">${role==='ct'?'CT’s face constraint and a Fourier derivative of cell B are different operators. Float32 roundoff accumulates.':'GLM divergence is measured with common cell-centered operators. It should not be equated to CT’s face constraint.'}</p>`);
      content+=card('Benchmark coverage','<p>Orszag–Tang tests shocked flow. The entropy-wave page adds an exact smooth solution. Shock tubes, magnetic-loop advection, rotors, current sheets and blasts are prepared for the GPU solvers.</p>'+links([['GPU test suite','gpu-suite.html']]),'next-step');
      if(role==='glm')content+=card('Earlier runs',img('../scripts/out/ot_density.png','Original Orszag–Tang SIMD preview')+links([['First 128³ SIMD animation','../scripts/out/ot_density.gif'],['Original 64² GLM + CT movie','../scripts/out/glm_ct_64/glm_ct_comparison.mp4'],['Original comparison still','../scripts/out/glm_ct_64/comparison.png']])+`<p class="small">Historical artifacts. The original SIMD run used cubic storage; the current comparison is a true 2D solver.</p>`);
    }else{
      content+=card('Reference progress',checks(['Athena++ installed and CPU build verified',`${checkN}² and ${refN}² runs complete to t = 0.5`,'21 actual states saved for the synchronized movie','Reference fields averaged to the comparison grid'])+links([['Athena++ installation & methodology','../scripts/ATHENA_REFERENCE.md'],[`${refN}² run manifest`,`../scripts/out/athena_${refN}/run_manifest.json`],[`${checkN}² run manifest`,`../scripts/out/athena_${checkN}/run_manifest.json`],['Dense-output movie run',`../scripts/out/athena_${movieRefN}_movie/run_manifest.json`],['Movie validation',`../scripts/out/athena_${movieRefN}_movie/movie_validation.json`]]));
      content+=card('Next reference refinement',`<p>${refN<1024?'1024² is configured, not run. This is the next refinement needed to test sensitivity of the current comparison.':'Assess reference-to-reference differences before deciding whether another refinement is needed.'}</p>`,'next-step');
      content+=card('Comparison library',`<div id="comparison-links"></div>`);
      content+=card('Analysis validation',stats([['Analytic / structural tests','8 passed'],['Spectra checked',D.ot.validation.spectra_checked],['Max. Parseval residual',Number(D.ot.validation.max_relative_parseval_residual).toExponential(2)]])+links([['Validation record','../scripts/out/spectral_study/analysis/validation.json'],['All norm measurements · CSV','../scripts/out/spectral_study/analysis/norms.csv'],['All spectra · CSV','../scripts/out/spectral_study/analysis/spectra.csv']]));
    }
    return column(role,role==='glm'?'Cell-centered B · Dedner cleaning':role==='ct'?'Face-centered B · constrained transport':'Athena++ · finite-resolution numerical reference',role==='truth'?'Provisional reference':'Run complete',content);
  }
  function updateOT(){
    const {n,field,time}=selection,run=D.ot.runs[n],t=D.ot.times[time],bounds=run.bounds[field];
    document.querySelectorAll('[data-resolution]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.resolution===n)));
    document.querySelector('#field').value=field;document.querySelector('#time').value=time;document.querySelector('#time-output').textContent=`t = ${t.toFixed(3)}`;
    document.querySelectorAll('[data-time-title]').forEach(e=>e.textContent=`t = ${t.toFixed(3)}`);
    document.querySelectorAll('[data-field-title]').forEach(e=>e.textContent=field==='density'?'Density ρ':'Magnetic magnitude |B|');
    document.querySelectorAll('[data-scale]').forEach(e=>e.classList.toggle('magnetic',field==='magnetic'));
    document.querySelectorAll('[data-scale-low]').forEach(e=>e.textContent=bounds[0].toFixed(3));document.querySelectorAll('[data-scale-high]').forEach(e=>e.textContent=bounds[1].toFixed(3));
    document.querySelector('#comparison-note-reference').textContent=n===refN?`${refN}² reference = equal resolution; not a finer-grid accuracy test.`:`Athena++ ${refN}² averaged to ${n}²; numerical reference remains provisional.`;
    const changeMovies=movieSelection!==`${n}-${field}`;
    if(changeMovies && transport)transport.pause();
    for(const role of roles){
      if(changeMovies){
        const video=document.querySelector(`#movie-${role}`);
        video.src=`media/ot-${role}-${n}-${field}.mp4`;
        video.poster=run.images[field][role][3];
        video.load();
        document.querySelector(`#movie-caption-${role}`).textContent=role==='truth'?`Athena++ ${movieRefN}² → ${n}² · ${field} · 21 saved states. Accuracy tables use ${refN}².`:`${n}² · ${field} · 21 saved states · same run as snapshots`;
      }
      const src=run.images[field][role][time],alt=`${labels[role]} · ${field} · ${n}² comparison grid · t=${t.toFixed(3)}`;
      const el=document.querySelector(`#snapshot-${role}`);el.src=src;el.alt=alt;
      const anchor=document.querySelector(`#snapshot-link-${role}`);anchor.href=src;anchor.dataset.caption=alt;
      document.querySelector(`#snapshot-caption-${role}`).textContent=role==='truth'?(n===refN?`Athena++ ${refN}² native grid · equal-resolution comparison`:`Athena++ ${refN}² → ${n}² conservative cell averages`):`${labels[role]} ${n}² · PLM / HLL / RK2 · Float32`;
      for(const kind of ['spectrum','energy']){
        const path=`media/${kind==='spectrum'?'spectra':'energies'}-${n}-${role}.png`;
        document.querySelector(`#${kind}-${role}`).src=path;document.querySelector(`#${kind}-link-${role}`).href=path;
      }
      if(role!=='truth'){
        const rows=D.ot.norms.filter(r=>+r.N===n&&+r.time===t&&r.method===role&&+r.reference_N===refN&&['density','magnetic'].includes(r.field));
        document.querySelector(`#norms-${role}`).innerHTML=`<table class="metrics-table"><thead><tr><th>Field</th><th>L₁</th><th>L₂</th><th>L∞</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.field==='density'?'Density':'Magnetic B'}</td>${['L1','L2','Linf'].map(k=>`<td>${(100*Number(r['relative_'+k])).toFixed(3)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
        document.querySelector(`#run-record-${role}`).innerHTML=stats([['Grid / output states',`${n}² / ${run.meta.frames}`],['Steps to t = 0.5',run.meta.steps.toLocaleString()],['Method stepping time',`${run.meta[role+'_step_seconds'].toFixed(2)} s`],['Maximum relative energy drift',run.meta.max_saved_relative_conservation_errors[role+'_energy'].toExponential(2)]]);
        document.querySelector(`#run-links-${role}`).innerHTML=links([['Run metadata',`../scripts/out/spectral_study/glm_ct_${n}/metadata.toml`],['Diagnostics · CSV',`../scripts/out/spectral_study/glm_ct_${n}/${role}_diagnostics.csv`]]);
      }
    }
    movieSelection=`${n}-${field}`;
    const s=run.truthStats[time];document.querySelector('#truth-statistics').innerHTML=stats([['Reference / comparison grid',`${refN}² / ${n}²`],['Reference density range',`${s.rhoMin.toFixed(3)}–${s.rhoMax.toFixed(3)}`],['Reference B rms',s.Brms.toFixed(4)],['Finer reference available',n<refN?`Yes · ${refN}²`:'No']]);
    document.querySelector('#reference-warning').textContent=n===refN?`At this resolution, the reported differences compare two codes on the same grid. A finer reference is needed.`:`There is no exact evolved Orszag–Tang solution here. The ${checkN}²–${refN}² difference checks reference sensitivity, not an exact error bound.`;
    const sensitivity=D.ot.norms.filter(r=>+r.N===n&&+r.time===t&&r.method==='reference_difference');
    document.querySelector('#reference-sensitivity').innerHTML=sensitivity.length?stats(sensitivity.map(r=>[`${checkN}² vs ${refN}² · ${r.field} L₁`,`${(100*+r.relative_L1).toFixed(3)}%`])):'<p class="small">No coarser-reference difference available on this grid.</p>';
    const base=`../scripts/out/spectral_study/analysis/N${n}/`;
    document.querySelector('#comparison-links').innerHTML=links([['Density & magnetic error spectra',base+'error_spectra.png'],['Power spectra · all methods',base+'power_spectra.png'],['Divergence · operators separated',base+'divergence_spectra.png'],['Energy histories',base+'energy_history.png'],['All resolutions & cost','../scripts/out/spectral_study/analysis/resolution_comparison.png'],['Density error maps',`../scripts/out/errors_athena_${refN}/density_errors.png`]]);
    try{history.replaceState(null,'',`?n=${n}&field=${field}&time=${time}`);}catch{}
  }

  // ---------------------------------------------------------------- GPU suite
  // Multi-code comparison: any two methods side by side against the reference, plus every method
  // in the head-to-head table. Hue = code (GLM, CT, AthenaK, RAMSES); variants share their code's hue.
  const G=D.gpu, fmt=(v,d=3)=>v==null||!isFinite(v)?'—':Math.abs(v)>=1e4||(Math.abs(v)<1e-3&&v!==0)?Number(v).toExponential(2):Number(v).toFixed(d);
  const statusText={complete:'Complete',failed:'Failed',incomplete:'Incomplete',missing:'Not run'};
  const pct=v=>v==null||!isFinite(v)?'—':(100*v).toFixed(v<.001?4:3);
  const methodLabel=m=>(G?.legend?.[m]?.label)||m, methodColor=m=>(G?.legend?.[m]?.color)||'#52514e', familyOf=m=>(G?.legend?.[m]?.family)||'ct';
  let gsel={test:G?.tests[0]?.name,key:null,field:null,scale:'shared',frame:null,a:null,b:null};
  const FIELD_TITLES={density:'Density ρ',magnetic:'Magnetic magnitude |B|',vorticity:'Vorticity ω_z',current:'Current density J_z'};
  let gpuTimer=null;
  function stopGpuPlayback(){
    if(gpuTimer){clearInterval(gpuTimer);gpuTimer=null;}
    const b=document.querySelector('#gplay');if(b){b.textContent='▶ Play frames';b.setAttribute('aria-pressed','false');}
  }
  function toggleGpuPlayback(){
    if(gpuTimer){stopGpuPlayback();return;}
    const run=G.runs?.[gsel.test]?.[gsel.key],count=run?.frames?.length||0;
    if(count<2)return;
    if(gsel.frame>=count-1)gsel.frame=0;
    const b=document.querySelector('#gplay');b.textContent='Ⅱ Pause frames';b.setAttribute('aria-pressed','true');
    gpuTimer=setInterval(()=>{
      const current=G.runs?.[gsel.test]?.[gsel.key],n=current?.frames?.length||0;
      if(n<2){stopGpuPlayback();return;}
      gsel.frame=(gsel.frame+1)%n;gpuUpdate();
    },500);
    gpuUpdate();
  }
  const preloaded=new Map();
  function preload(src){if(!src)return Promise.resolve();if(!preloaded.has(src)){const im=new Image();preloaded.set(src,new Promise(r=>{im.onload=im.onerror=()=>r();}));im.src=src;}return preloaded.get(src);}
  function setShot(box,src,alt,empty){
    const link=box.querySelector('a.image-link'),im=link&&link.querySelector('img');
    if(!src){box.dataset.want='';box.innerHTML=empty;return;}
    if(!im){box.innerHTML=img(src,alt).replace('media-image','snapshot-image').replace(' loading="lazy"','');box.dataset.want=src;return;}
    box.dataset.want=src;
    preload(src).then(()=>{if(box.dataset.want!==src)return;im.src=src;im.alt=alt;link.href=src;link.dataset.caption=alt;});
  }
  let extraKey='';
  function gpuRuns(name){return Object.entries((G.runs||{})[name]||{}).sort((a,b)=>(a[1].smoke-b[1].smoke)||(a[1].N-b[1].N)||(a[1].seed-b[1].seed));}
  function matrixColumns(){return ['8³ smoke',...[...new Set(G.tests.flatMap(t=>t.grids))].sort((a,b)=>a-b).map(n=>`${n}³`)];}
  // One dot per code: filled when every variant of that code completed, ring when any failed.
  function familyDots(t,run){
    return Object.entries(G.codeFamilies).filter(([f])=>t.expectedMethods.some(m=>familyOf(m)===f)).map(([f,info])=>{
      const ms=t.expectedMethods.filter(m=>familyOf(m)===f),st=ms.map(m=>run?.summaries?.[m]?.status||'missing');
      const state=st.every(x=>x==='complete')?'complete':st.some(x=>x==='failed')?'failed':st.some(x=>x!=='missing')?'incomplete':'missing';
      return `<i class="dot ${state}" style="--dot:${info.color}" title="${esc(info.label)}: ${ms.map((m,i)=>`${methodLabel(m)} ${statusText[st[i]]}`).join(', ')}"></i>`;
    }).join('');
  }
  function matrixCells(t){
    const runs=gpuRuns(t.name);
    return matrixColumns().map(col=>{
      if(col!=='8³ smoke'&&!t.grids.some(n=>`${n}³`===col))return '<td class="not-configured" title="Grid not in this config">·</td>';
      const hits=runs.filter(([,r])=>r.smoke?col==='8³ smoke':col===`${r.N}³`);
      if(!hits.length)return `<td><span class="dots" title="Not run yet">${familyDots(t,null)}</span></td>`;
      return `<td>${hits.map(([key,r])=>`<button type="button" class="dots" data-pick="${esc(t.name)}|${esc(key)}">${familyDots(t,r)}</button>`).join('')}</td>`;
    }).join('');
  }
  function gpuMatrix(){
    const cols=matrixColumns();
    const rows=G.families.map(f=>{const tests=G.tests.filter(t=>t.family===f);if(!tests.length)return '';
      return `<tr class="family-row"><th colspan="${cols.length+2}">${esc(f)}</th></tr>`+tests.map(t=>`<tr data-row="${esc(t.name)}"><td><button type="button" class="test-link" data-pick="${esc(t.name)}|">${esc(t.name)}</button>${t.divb?'<span class="divb-flag" title="∇·B treatments can differ">∇·B</span>':''}</td><td class="source-cell">${esc(t.source)}</td>${matrixCells(t)}</tr>`).join('');}).join('');
    const empty=!G.groupsFound?`<p class="notice">No Sherlock returns found in <code>${esc(G.source)}</code>. Extract downloaded archives there, then run <code>.venv-movie/bin/python scripts/build_progress_site.py --gpu-only</code>.</p>`:'';
    const legend=Object.values(G.codeFamilies).map(f=>`<i class="dot complete" style="--dot:${f.color}"></i>${esc(f.label)}${f.device==='cpu'?' (CPU)':''}`).join(' ');
    return `<details class="suite-status" open><summary><span class="control-label">Suite status</span><span class="legend">${legend} · filled = all variants complete · ring = a failure · dashed = not run · ${G.groupsFound} result sets · ${G.references.length} reference runs</span></summary>${empty}<div class="matrix-scroll"><table class="suite-matrix"><thead><tr><th>Test</th><th>Origin</th>${cols.map(c=>`<th>${c==='8³ smoke'?'Smoke 8³':c}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></details>`;
  }
  function gpuColumn(slot){
    const truth=slot==='truth';
    const chooser=truth?'':`<select class="method-select field-select" data-slot="${slot}" aria-label="Method for column ${slot.toUpperCase()}"></select>`;
    return `<section class="column method-col ${truth?'truth':''}" data-col="${slot}" aria-label="${truth?'Reference':'Method '+slot.toUpperCase()}"><div class="column-head"><div class="column-title"><h2>${truth?'The Truth':`<span class="swatch" data-swatch></span>${chooser}`}</h2><span class="status-pill"></span></div><p data-subtitle></p></div>`+
      `<section class="card snapshot-card"><div class="snapshot-heading"><span data-field-title>Density</span><span data-time-title></span></div><div data-shot></div><div class="snapshot-foot"><div class="scale" data-scale></div><div class="scale-labels"><span data-scale-low></span><span data-plane-note></span><span data-scale-high></span></div><p class="snapshot-caption" data-shot-caption></p></div></section><div data-cards></div>${note(slot)}</section>`;
  }
  function defaults(run){
    const ms=run?run.methods:[],pick=list=>list.find(m=>ms.includes(m));
    if(!ms.includes(gsel.a))gsel.a=pick(['glm_hll16','glm','glm_hlld','glm_ppm','glm_constch'])||ms[0]||null;
    if(!ms.includes(gsel.b)||gsel.b===gsel.a)gsel.b=pick(['athenak_best','ct','ramses_best','athenak_matched'].filter(m=>m!==gsel.a))||ms.find(m=>m!==gsel.a)||null;
  }
  function gpuUpdate(){
    const t=G.tests.find(x=>x.name===gsel.test),runs=gpuRuns(t.name);
    if(!runs.some(([k])=>k===gsel.key)){const full=runs.filter(([,r])=>!r.smoke),top=Math.max(...full.map(([,r])=>r.N));gsel.key=(full.find(([,r])=>r.N===top)||runs[0]||[null])[0];}
    if(!FIELD_TITLES[gsel.field])gsel.field=t.defaultField||'density';
    const run=gsel.key?G.runs[t.name][gsel.key]:null,frames=run?run.frames:[];
    if(gsel.frame==null||gsel.frame>=frames.length)gsel.frame=Math.max(0,frames.length-1);
    defaults(run);
    document.querySelector('#gtest').value=t.name;document.querySelector('#gfield').value=gsel.field;document.querySelector('#gscale').value=gsel.scale;
    document.querySelector('#ggrid').innerHTML=runs.length?runs.map(([k,r])=>`<button type="button" data-gkey="${esc(k)}" aria-pressed="${k===gsel.key}">${esc(r.label)}</button>`).join(''):t.grids.map(n=>`<button type="button" disabled>${n}³</button>`).join('');
    const slider=document.querySelector('#gframe');slider.max=Math.max(0,frames.length-1);slider.value=gsel.frame;slider.disabled=frames.length<2;
    const play=document.querySelector('#gplay');play.disabled=frames.length<2;play.textContent=gpuTimer?'Ⅱ Pause frames':'▶ Play frames';play.setAttribute('aria-pressed',gpuTimer?'true':'false');
    const f=frames[gsel.frame];
    document.querySelector('#gframe-out').textContent=f?`t = ${fmt(f.time,4)}`:'no output';
    document.querySelector('#gframe-ticks').innerHTML=frames.length?frames.map(x=>`<span>${fmt(x.time,3)}</span>`).join(''):'<span>awaiting runs</span>';
    document.querySelector('#gnote').innerHTML=`<span>${esc(t.focus)}</span><strong>${esc(t.adaptation)}</strong>`;
    document.querySelectorAll('[data-row]').forEach(r=>r.classList.toggle('selected',r.dataset.row===t.name));
    document.querySelector('#intro h1').textContent=`Multi-code suite · ${t.name}`;
    const perOut=gsel.scale==='output'&&run&&run.boundsPerOutput,kind=run?.scales?.[gsel.field]||'linear';
    const bounds=run?(perOut?run.boundsPerOutput[gsel.field]?.[gsel.frame]:run.bounds[gsel.field]):null,imageSet=run?(perOut?run.imagesPerOutput:run.images):null;
    for(const slot of ['a','b','truth']){
      const col=document.querySelector(`[data-col="${slot}"]`),m=slot==='truth'?'truth':gsel[slot];
      col.style.setProperty('--accent',slot==='truth'?'var(--truth)':methodColor(m));
      if(slot!=='truth'){
        const sel=col.querySelector('select');sel.innerHTML=(run?run.methods:t.expectedMethods).map(x=>`<option value="${esc(x)}">${esc(methodLabel(x))}</option>`).join('');
        if(m)sel.value=m;else sel.value='';
        col.querySelector('[data-swatch]').style.background=methodColor(m);
      }
      col.querySelector('[data-field-title]').textContent=FIELD_TITLES[gsel.field]+(kind==='log'?' · log scale':'');
      col.querySelector('[data-time-title]').textContent=f?`t = ${fmt(f.time,4)}`:'';
      col.querySelector('[data-scale]').classList.toggle('magnetic',gsel.field==='magnetic');col.querySelector('[data-scale]').classList.toggle('vorticity',gsel.field==='vorticity');col.querySelector('[data-scale]').classList.toggle('current',gsel.field==='current');
      col.querySelector('[data-scale-low]').textContent=bounds?fmt(bounds[0]):'';col.querySelector('[data-scale-high]').textContent=bounds?fmt(bounds[1]):'';
      col.querySelector('[data-plane-note]').textContent=`${perOut?'Scale of this output':'Scale shared over time'} · ${run&&!run.planar?'midplane z = ½':'x, y ∈ [0, 1]'}`;
      const src=run&&f&&m?imageSet[gsel.field]?.[m]?.[gsel.frame]:null,alt=`${slot==='truth'?'Reference':methodLabel(m)} · ${t.name} · ${run?run.label:''} · ${gsel.field} · t=${f?fmt(f.time,4):''}`;
      setShot(col.querySelector('[data-shot]'),src,alt,`<div class="snapshot-empty"><div class="empty-symbol">—</div><p>${!run?'Awaiting Sherlock runs':slot==='truth'?(f&&f.reference===null?'No reference at this output':'No reference for this test'):'No saved state at this output'}</p></div>`);
      if(run&&m)(imageSet[gsel.field]?.[m]||[]).forEach(preload);
      const sm=run&&slot!=='truth'?run.summaries[m]:null;
      col.querySelector('[data-shot-caption]').textContent=!run?'':slot==='truth'?(f&&f.reference?f.reference:t.reference):sm?`${run.label} · ${sm.device}${run.synthetic?' · NOT A SHERLOCK MEASUREMENT':''}`:'';
      col.querySelector('.status-pill').textContent=slot==='truth'?(f&&f.reference?'Reference available':'No reference'):sm?statusText[sm.status]:(run?'Not run':'Awaiting runs');
      col.querySelector('[data-subtitle]').textContent=slot==='truth'?'Exact solution, AthenaK high-resolution reference or Athena++':sm?sm.solver:'Choose a method';
      col.querySelector('[data-cards]').innerHTML=slot==='truth'?truthCards(t,run,f):methodCards(m,t,run,f);
    }
    const key=[t.name,gsel.key,gsel.field,gsel.a,gsel.b].join('|');
    if(key!==extraKey){extraKey=key;document.querySelector('#extra').innerHTML=movieCard(t)+headToHead(t,run);}
    initNotes(`gpu-suite:${t.name}`);
    try{history.replaceState(null,'',`?test=${encodeURIComponent(t.name)}${gsel.key?`&run=${encodeURIComponent(gsel.key)}`:''}&field=${gsel.field}&scale=${gsel.scale}&frame=${gsel.frame}${gsel.a?`&a=${gsel.a}`:''}${gsel.b?`&b=${gsel.b}`:''}`);}catch{}
  }
  function methodCards(m,t,run,f){
    if(!run)return card('Awaiting runs',`<p>On Sherlock: <code>bash tools/submit.sh ${esc(t.name)}</code> for GLM and CT, and <code>--code athenak:best</code> etc. for external codes. Download, extract, then refresh this page’s data.</p>`,'empty-card');
    const s=m&&run.summaries[m];let html='';
    if(run.synthetic)html+=`<p class="notice">Part of this result set is labelled as a fixture or local test. It is not a Sherlock measurement.</p>`;
    if(!s)return html+card('Not run at this grid',`<p>${esc(methodLabel(m))} has no result for ${esc(run.label)}.</p>`,'empty-card');
    const e=f&&f.errors[m];
    html+=card('Difference from the reference',e?`<table class="metrics-table"><thead><tr><th>Field</th><th>L₁ %</th><th>L₂ %</th><th>L∞ %</th></tr></thead><tbody>${['density','magnetic'].map(k=>`<tr><td>${k==='density'?'Density':'Magnetic B'}</td>${['L1','L2','Linf'].map(n=>`<td>${pct(e[k]['relative_'+n])}</td>`).join('')}</tr>`).join('')}</tbody></table><p class="small">Relative norms at this output. Magnetic errors use the full vector.</p>`:`<p>${f&&f.reference?'No saved state at this output.':'No reference at this output; compare columns and the histories below.'}</p>`);
    html+=card('Run record',(s.error?`<p class="notice">${esc(s.error)}</p>`:'')+stats([['Status',`${statusText[s.status]} · ${s.repeatsComplete}/${s.repeatsTotal} repeats`],['Steps / time reached',`${s.steps??'—'} / ${fmt(s.timeReached,4)}`],['Median solve time',s.solveMedian==null?'—':`${fmt(s.solveMedian,2)} s (IQR ${fmt(s.solveIQR[0],2)}–${fmt(s.solveIQR[1],2)})`],['Cost unit',s.costUnit],['Median throughput',s.rateMedian==null?'—':`${fmt(s.rateMedian,1)} Mcell-updates/s`],['Precision',s.precision||'—'],['Device',s.device]])+links(s.links)+`<p class="small">Timing medians use repeats 2–${s.repeatsTotal} (repeat 1 writes snapshots). Throughput counts N³ cells, also for planar tests.</p>`);
    html+=card('Health over the run',stats([['Max energy / mass drift',`${fmt(s.maxEnergyDrift)} / ${fmt(s.maxMassDrift)}`],[s.divergenceKind,m.startsWith('athenak')||m.startsWith('ramses')?'not saved':fmt(s.maxDivergence)],['Max |Bz|',fmt(s.maxBz)],['Min unfloored pressure',fmt(s.minPressure)],['Max nonpositive-pressure cells',s.maxNonpositiveCells==null?'—':`${s.maxNonpositiveCells}${s.firstNonpositivePressure!=null?` · first at t = ${fmt(s.firstNonpositivePressure,4)}`:''}`]]));
    return html;
  }
  function truthCards(t,run,f){
    let html=card('What this test measures',`<p>${esc(t.focus)}</p><p class="small">${esc(t.adaptation)}</p>`+stats([['Origin',t.source],['γ · end time',`${t.gamma} · ${t.tEnd}`],['Outputs · grids',`${t.frames} · ${t.grids.map(n=>n+'³').join(', ')}${t.seeds.length>1?` · seeds ${t.seeds.join(', ')}`:''}`],['Geometry',t.planar?'planar (z invariant), N³ storage':'full 3D'],['Discriminates ∇·B treatment',t.divb?'yes':'no — control test'],['Reference run',t.referenceN?`AthenaK ${t.referenceN}${t.planar?'²':'³'}`:'exact or none'],['Methods planned',String(t.expectedMethods.length)],['Setup and results from',t.bundle==='sherlock_rerun'?'sherlock_rerun · corrected setup, rerun 2026-09-28':t.bundle||'—']]));
    const sens=f&&f.sensitivity;
    html+=card('The reference',`<p>${esc(t.reference)}</p>`+(run&&f?`<p class="small">${f.reference?`At this output: ${esc(f.reference)}.`:'No reference exists at this output.'}</p>`:'')+(sens?stats([['Athena++ vs AthenaK reference · density L₁',pct(sens.density)+' %'],['· magnetic L₁',pct(sens.magnetic)+' %']])+'<p class="small">Reference sensitivity: errors smaller than this are not resolved by the reference.</p>':'')+(t.reference.startsWith('none')&&!t.referenceN?'<p class="small">Without a reference, differences between columns are method differences, not errors.</p>':''));
    return html;
  }
  function movieCard(t){
    const mv=(G.movies||{})[t.name];if(!mv)return '';
    const fields=Object.keys(mv.fields),field=fields.includes(gsel.field)?gsel.field:fields.includes('vorticity')?'vorticity':fields[0];
    const dense=mv.source&&mv.source.includes('movies');
    return `<section class="card head-to-head movie-suite"><h3>Movie · GLM | CT prototype | ${esc(mv.truth)}</h3><div class="segmented" role="group" aria-label="Movie field">${fields.map(f=>`<button type="button" data-movie-field="${esc(f)}" aria-pressed="${f===field}">${esc(FIELD_TITLES[f]||f)}</button>`).join('')}</div><video controls loop muted playsinline preload="metadata" src="${esc(mv.fields[field])}"></video><p class="small">${mv.frames} saved outputs to t = ${fmt(mv.tEnd,3)} at ${mv.grid}², one colour scale for all three panels${dense?'. Dense-output movie run (not timed).':'. Preview built from the accuracy runs; the dense-output movie runs replace it.'}</p></section>`;
  }
  function headToHead(t,run){
    const scaling=G.scaling[t.name];let rows='';
    if(run)rows=run.methods.map(m=>{const x=run.summaries[m],last=run.frames.length?run.frames[run.frames.length-1].errors[m]:null;return `<tr><td><span class="swatch" style="background:${methodColor(m)}"></span>${esc(methodLabel(m))}</td><td>${statusText[x.status]}</td><td>${x.solveMedian==null?'—':fmt(x.solveMedian,2)}</td><td>${esc(x.costUnit.split(' ')[0])}</td><td>${fmt(x.rateMedian,1)}</td><td>${last?pct(last.density.relative_L1):'—'}</td><td>${last?pct(last.magnetic.relative_L1):'—'}</td><td>${fmt(x.maxBz)}</td><td>${x.maxNonpositiveCells??'—'}</td><td>${esc(x.precision||'—')}</td></tr>`;}).join('');
    return `<section class="card head-to-head"><h3>Every method${run?` · ${esc(run.label)}`:''}</h3>${run?`<div class="table-scroll"><table class="metrics-table"><thead><tr><th>Method</th><th>Status</th><th>Median solve</th><th>Unit</th><th>Mcell-upd/s</th><th>Final ρ L₁ %</th><th>Final B L₁ %</th><th>Max |Bz|</th><th>p≤0 cells</th><th>Precision</th></tr></thead><tbody>${rows}</tbody></table></div><div class="figure-row">${img(run.history,`${t.name} ${run.label} time histories, every method`)}</div>${(run.extras||[]).map(x=>`<div class="figure-row">${img(x.image,x.caption)}</div><p class="small">${esc(x.caption)}</p>`).join('')}`:'<p>No results for this test yet. The status matrix fills in as archives are extracted and the page is rebuilt.</p>'}${scaling?`<div class="figure-row">${img(scaling,`${t.name}: accuracy vs grid, accuracy vs GPU cost, throughput`)}</div><p class="small">Accuracy uses the final output of each grid where a reference exists. GPU and CPU seconds are not comparable, so RAMSES is left out of the cost panel.</p>`:''}</section>`;
  }
  if(test.kind==='gpu'){
    try{const p=new URLSearchParams(location.search);if(G.tests.some(t=>t.name===p.get('test')))gsel.test=p.get('test');if(p.get('run'))gsel.key=p.get('run');if(['density','magnetic','vorticity','current'].includes(p.get('field')))gsel.field=p.get('field');if(p.get('scale')==='output')gsel.scale='output';if(p.has('frame'))gsel.frame=+p.get('frame');if(p.get('a'))gsel.a=p.get('a');if(p.get('b'))gsel.b=p.get('b');}catch{}
    const options=G.families.map(f=>{const tests=G.tests.filter(t=>t.family===f);return tests.length?`<optgroup label="${esc(f)}">${tests.map(t=>`<option value="${esc(t.name)}">${esc(t.name)}</option>`).join('')}</optgroup>`:'';}).join('');
    document.querySelector('#controls').innerHTML=gpuMatrix()+`<div class="toolbar"><div class="control-group"><label class="control-label" for="gtest">Test</label><select id="gtest" class="field-select">${options}</select></div><div class="control-group"><span class="control-label" id="ggrid-label">Grid</span><div class="segmented" id="ggrid" role="group" aria-labelledby="ggrid-label"></div></div><div class="control-group"><label class="control-label" for="gfield">Field</label><select id="gfield" class="field-select"><option value="density">Density ρ</option><option value="magnetic">Magnetic |B|</option><option value="vorticity">Vorticity ω_z</option><option value="current">Current density J_z</option></select></div><div class="control-group"><label class="control-label" for="gscale">Colour scale</label><select id="gscale" class="field-select"><option value="shared">Shared over time</option><option value="output">Each output</option></select></div><div class="time-control"><div class="time-label"><label class="control-label" for="gframe">Saved output</label><output id="gframe-out" for="gframe"></output></div><input id="gframe" type="range" min="0" max="0" step="1" value="0" aria-label="Saved output"><div class="time-ticks" id="gframe-ticks"></div></div><button id="gplay" type="button" class="quiet-button" aria-pressed="false">▶ Play frames</button></div><div class="comparison-note" id="gnote"></div><p class="mobile-hint small">Play advances the same saved output in all three columns. Scroll sideways to compare them.</p>`;
    document.querySelector('#board').innerHTML=['a','b','truth'].map(gpuColumn).join('');
    document.querySelector('#gtest').addEventListener('change',e=>{stopGpuPlayback();gsel={...gsel,test:e.target.value,key:null,frame:null,field:null};gpuUpdate();});
    document.querySelector('#gfield').addEventListener('change',e=>{gsel.field=e.target.value;gpuUpdate();});
    document.querySelector('#gscale').addEventListener('change',e=>{gsel.scale=e.target.value;gpuUpdate();});
    document.querySelector('#extra').addEventListener('click',e=>{const b=e.target.closest('[data-movie-field]');if(b){gsel.field=b.dataset.movieField;gpuUpdate();}});
    document.querySelector('#gframe').addEventListener('input',e=>{stopGpuPlayback();gsel.frame=+e.target.value;gpuUpdate();});
    document.querySelector('#gplay').addEventListener('click',toggleGpuPlayback);
    document.querySelector('#ggrid').addEventListener('click',e=>{const b=e.target.closest('[data-gkey]');if(b){stopGpuPlayback();gsel.key=b.dataset.gkey;gsel.frame=null;gpuUpdate();}});
    document.querySelector('#board').addEventListener('change',e=>{const s=e.target.closest('.method-select');if(s){gsel[s.dataset.slot]=s.value;gpuUpdate();}});
    document.querySelector('#controls').addEventListener('click',e=>{const b=e.target.closest('[data-pick]');if(!b)return;stopGpuPlayback();const [name,key]=b.dataset.pick.split('|');gsel={...gsel,test:name,key:key||null,frame:null,field:name===gsel.test?gsel.field:null};gpuUpdate();document.querySelector('#gtest').scrollIntoView({block:'nearest',behavior:'smooth'});});
    gpuUpdate();
  }
  if(test.kind==='ot'){
    document.querySelector('#controls').innerHTML=`<div class="toolbar"><div class="control-group"><span class="control-label" id="res-label">Grid</span><div class="segmented" role="group" aria-labelledby="res-label">${[64,128,256,512].map(n=>`<button type="button" data-resolution="${n}" aria-pressed="false">${n}²</button>`).join('')}</div></div><div class="control-group"><label class="control-label" for="field">Field</label><select id="field" class="field-select"><option value="density">Density ρ</option><option value="magnetic">Magnetic |B|</option></select></div><div class="time-control"><div class="time-label"><label class="control-label" for="time">Saved checkpoint</label><output id="time-output" for="time"></output></div><input id="time" type="range" min="0" max="3" step="1" value="3" aria-label="Saved checkpoint"><div class="time-ticks"><span>0</span><span>0.1</span><span>0.25</span><span>0.5</span></div></div></div><div class="comparison-note"><span>Snapshots share a color scale at each resolution. Movies follow the selected grid and field; changing either resets playback.</span><strong id="comparison-note-reference"></strong></div><p class="mobile-hint small">Scroll sideways to compare all three columns.</p>`;
    document.querySelector('#board').innerHTML=roles.map(otColumn).join('');updateOT();
    document.querySelectorAll('[data-resolution]').forEach(b=>b.addEventListener('click',()=>{selection.n=+b.dataset.resolution;updateOT();}));
    document.querySelector('#field').addEventListener('change',e=>{selection.field=e.target.value;updateOT();});
    document.querySelector('#time').addEventListener('input',e=>{selection.time=+e.target.value;updateOT();});
    const videos=roles.map(role=>document.querySelector(`#movie-${role}`));
    videos.forEach(video=>{video.muted=true;});
    transport=new SynchronizedVideos(videos,(playing,message)=>{
      document.querySelectorAll('[data-play-all]').forEach(button=>button.textContent=playing?'Ⅱ Pause all three':'▶ Play all three');
      document.querySelectorAll('.movie-status').forEach(el=>el.textContent=message);
    });
    document.querySelectorAll('[data-play-all]').forEach(button=>button.addEventListener('click',()=>transport.toggle()));
  }else if(test.kind!=='gpu'){
    document.querySelector('#controls').innerHTML='<div class="comparison-note"><span>The same three-column notebook is used for every test. Hydro-only runs are labeled explicitly.</span><strong>“The Truth” records the available reference and its limits.</strong></div>';
    document.querySelector('#board').innerHTML=test.columns.map((c,i)=>column(roles[i],c.subtitle,c.status,c.cards.map(item=>{
      let body=item.empty?'<div class="empty-symbol">—</div>':'';
      if(item.formula)body+=`<div class="formula">${esc(item.formula)}</div>`;
      if(item.video)body+=`<video controls playsinline preload="metadata" poster="${esc(item.poster)}" style="width:100%;border-radius:5px" aria-label="${esc(item.title)}"><source src="${esc(item.video)}" type="video/mp4"></video>`;
      if(item.image)body+=img(item.image,item.caption||item.title,'card-image');
      if(item.caption)body+=`<div class="media-caption">${esc(item.caption)}</div>`;
      if(item.stats)body+=stats(item.stats);if(item.items)body+=checks(item.items);
      if(item.text)body+=`<p>${esc(item.text)}</p>`;if(item.links)body+=links(item.links);
      return card(item.title,body,item.empty?'empty-card':'');
    }).join(''))).join('');
  }

  if(test.kind==='generic'){
    const videos=Array.from(document.querySelectorAll('#board video'));
    if(videos.length>1){videos.forEach(v=>v.muted=true);transport=new SynchronizedVideos(videos);}
  }

  // These boxes kept notes in this browser only. Notes now live on the private notes page (saved to GitHub), which copies
  // in whatever a box holds when it is next unlocked in this browser; a box that holds text stays visible, read-only.
  function initNotes(scope){
    document.querySelectorAll('textarea[data-note]').forEach(el=>{
      const key=`mhd-notes:${scope}:${el.dataset.note}`,state=document.querySelector(`#note-state-${el.dataset.note}`),card=el.closest('.notes-card');
      let text='';try{text=localStorage.getItem(key)||'';}catch{}
      if(!text.trim()){if(card)card.remove();return;}
      el.value=text;el.readOnly=true;
      if(state)state.textContent='Kept in this browser only. Copied into your notes page the next time you unlock it there; new notes go there.';
    });
  }
  if(test.kind!=='gpu')initNotes(test.id);
  const dialog=document.querySelector('#image-viewer');
  document.addEventListener('click',e=>{const anchor=e.target.closest('a[data-zoom]');if(!anchor)return;e.preventDefault();const image=dialog.querySelector('img');image.src=anchor.href;image.alt=anchor.dataset.caption||anchor.querySelector('img')?.alt||'Research figure';dialog.querySelector('p').textContent=image.alt;dialog.querySelector('a').href=anchor.href;dialog.showModal();});
  dialog.querySelector('button').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
})();
