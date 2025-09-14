function hasSrc(v) {
  return !!v.getAttribute('src') || !!v.currentSrc;
}

function hydrateVideo(v) {
  if (!hasSrc(v)) {
    const url = v.dataset && v.dataset.src;
    if (url) {
      v.setAttribute('src', url);
      // אם את מעדיפה: v.src = url;
      v.load();
    }
  }
}

function activateSlide(slideEl) {
  const vids = slideEl.querySelectorAll('video');
  vids.forEach(v => {
    // חשוב במובייל: לשים מאפיינים לפני play
    v.muted = true;
    v.playsInline = true;
    hydrateVideo(v);
  });
  vids.forEach(v => {
    v.play().catch(err => {
      // בשקט: ייתכן שדורש אינטראקציה
      // console.debug('play blocked', err);
    });
  });
}

function deactivateSlide(slideEl, {unload=false} = {}) {
  const vids = slideEl.querySelectorAll('video');
  vids.forEach(v => {
    if (!v.paused) v.pause();
    if (unload && hasSrc(v)) {
      // משחרר דקודר/זיכרון—שימושי במובייל כשיש הרבה וידאו
      const t = v.currentTime; // אם תרצי לשמר “מצב”
      v.removeAttribute('src');
      v.load();
      // אפשר לשמור currentTime בצד אם תרצי לחזור לאותה נקודה
      // v.dataset.resumeTime = t;
    }
  });
}

function getActiveItem(carouselEl) {
  return carouselEl.querySelector('.carousel-item.is-active');
}



// Main application JavaScript
document.addEventListener('DOMContentLoaded', () => {
  // ── Bulma carousel initialization ───────────────────────────────────────
  if (window.bulmaCarousel) {
    const carousels = bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 1,
      infinite: true,
      autoplay: false,
    });

    carousels.forEach(carousel => {
  carousel.on('before:show', () => {
    // עוצרים את כל הווידאו בשקופית הנוכחית,
    // ואם זו קרוסלת מובייל “כבדה” – אפשר גם unload
    const current = getActiveItem(carousel.element);
    if (current) {
      const isUsecase4 = !!carousel.element.closest('#usecase4');
      // ב-usecase4 יש לך דרישות ייחודיות—כאן רק עוצרים.
      // בשאר: אם יש הרבה וידאו/מכשירים חלשים, שקלי unload:true
      deactivateSlide(current, {unload: false});
    }
  });

  carousel.on('after:show', (e) => {
    const active = getActiveItem(e.target);
    if (active) {
      // לפני play – לשים src מתוך data-src ולקרוא load()
      activateSlide(active);
      // אם שמרת resumeTime קודם:
      // active.querySelectorAll('video').forEach(v => {
      //   if (v.dataset.resumeTime) {
      //     v.currentTime = parseFloat(v.dataset.resumeTime);
      //     delete v.dataset.resumeTime;
      //   }
      // });
    }
  });

      // Manually trigger playback for the initial active slide in each carousel
  const initial = getActiveItem(carousel.element);
    if (initial) {
      activateSlide(initial);
      // טיפול ב-autoplay שנחסם – בדיוק כמו שהיה לך:
      initial.querySelectorAll('video').forEach(video => {
        const p = video.play();
        if (p && p.catch) {
          p.catch(() => {
            const container = video.closest('.videos-container') || initial;
            container.addEventListener('click', () => {
              container.querySelectorAll('video').forEach(v => v.play());
            }, { once: true });
          });
        }
      });
    }
  });
  }


  // ── Pairwise sync per .videos-container (first two <video>) ──────────
  document.querySelectorAll('.videos-container').forEach(box => {
    const vids = box.querySelectorAll('video');
    if (vids.length >= 2) {
      syncPair(vids[0], vids[1]);
    }
  });

  // ── Control-signal timelines ────────────────
  initializeControlSignals();
});

/* ==================== Tight video sync ==================== */
function syncPair(a, b) {
  let master = a, follower = b;
  let guard = false;
  const MAX_DRIFT = 0.05; // 50 ms

  const setMaster = (m, f) => { master = m; follower = f; };
  ['play','pause','seeking','seeked','ratechange','volumechange'].forEach(evt => {
    a.addEventListener(evt, () => setMaster(a, b));
    b.addEventListener(evt, () => setMaster(b, a));
  });

  // Mirror play/pause without ping-pong
  a.addEventListener('play',  () => mirror(() => b.play()));
  b.addEventListener('play',  () => mirror(() => a.play()));
  a.addEventListener('pause', () => mirror(() => !b.paused && b.pause()));
  b.addEventListener('pause', () => mirror(() => !a.paused && a.pause()));

  // Align once both know metadata
  const align = () => {
    if (a.readyState >= 1 && b.readyState >= 1) {
      mirror(() => { b.currentTime = a.currentTime; });
    }
  };
  a.addEventListener('loadedmetadata', align);
  b.addEventListener('loadedmetadata', align);

  // Keep drift small using RAF (smoother than 'timeupdate')
  (function tick(){
    if (!master.paused && !master.seeking && !follower.seeking &&
        master.readyState >= 2 && follower.readyState >= 2) {
      const drift = master.currentTime - follower.currentTime;
      if (Math.abs(drift) > MAX_DRIFT) {
        mirror(() => { follower.currentTime = master.currentTime; });
      }
    }
    requestAnimationFrame(tick);
  })();

  // Loop together (safety)
  a.addEventListener('ended', () => mirror(() => { a.currentTime = b.currentTime = 0; a.play(); b.play(); }));
  b.addEventListener('ended', () => mirror(() => { a.currentTime = b.currentTime = 0; a.play(); b.play(); }));

  function mirror(fn){
    if (guard) return;
    guard = true;
    try { fn(); } finally { guard = false; }
  }
}

/* ==================== Control signal timelines ==================== */
function initializeControlSignals() {
  // 20-frame control vectors
  const controlVectors = {
    // Use Case 1: Single Object Temporal Reordering
    'usecase1-1': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    'usecase1-2': [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    'usecase1-3': [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
    'usecase1-4': [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    'usecase1-5': [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],

    // Use Case 2: Multiple Object Temporal Reordering
    'usecase2-1': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
    'usecase2-2': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
    'usecase2-3': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
    'usecase2-4': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
    'usecase2-5': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],

    // Use Case 3: Action-Aligned Generation
    'usecase3-1': [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'usecase3-2': [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'usecase3-3': [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
    'usecase3-4': [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
    'usecase3-5': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
  };

  // For each usecase key, connect video+SVG if present
  Object.keys(controlVectors).forEach(key => {
    const video = document.getElementById(`vid-${key}`);
    const svg   = document.getElementById(`plot-${key}`);
    if (video && svg) {
      initTimeline({ video, svg, vec: controlVectors[key], color: '#FF7F0E' });
    }
  });
}

/* ==== Timeline initializer ==== */
function initTimeline({ video, svg, vec, color = '#FF7F0E' }) {
  const N = vec.length;
  const W = 900, H = 120, marginX = 30;
  const y0 = 90, y1 = 30;

  function xAt(i){ return marginX + i * ((W - 2*marginX) / (N - 1)); }
  function yFor(v){ return v ? y1 : y0; }

  // build points
  const pts = Array.from({length:N}, (_,i)=>[xAt(i), yFor(vec[i])]);

  // polyline
  const path = make('polyline', { fill:'none', stroke:color, 'stroke-width':8,
    points: pts.map(p=>p.join(',')).join(' ') });
  svg.appendChild(path);

  // dots
  for (const [x,y] of pts) {
    svg.appendChild(make('circle', { cx:x, cy:y, r:10, fill:color }));
  }

  // moving marker + dashed cursor
  const marker = make('circle', { r:11, fill:color, stroke:'white', 'stroke-width':3 });
  const cursor = make('line', { y1:10, y2:H-10, stroke:color, 'stroke-width':3, 'stroke-dasharray':'4 4' });
  svg.appendChild(cursor);
  svg.appendChild(marker);

  let stepDuration = 0.25; // fallback; replaced once metadata loads

  function updateByTime(t){
    const exact = t / stepDuration;
    const i = Math.min(N-1, Math.floor(exact));
    const frac = Math.max(0, Math.min(1, exact - i));
    const x0 = pts[i][0], y0p = pts[i][1];
    const x1 = (i < N-1) ? pts[i+1][0] : pts[i][0];
    const y1p = (i < N-1) ? pts[i+1][1] : pts[i][1];
    const xm = x0 + (x1 - x0) * frac;
    const ym = y0p + (y1p - y0p) * frac;
    marker.setAttribute('cx', xm); marker.setAttribute('cy', ym);
    cursor.setAttribute('x1', xm); cursor.setAttribute('x2', xm);
  }

  // events
  video.addEventListener('loadedmetadata', () => {
    stepDuration = video.duration / N;
    updateByTime(0);
  });
  video.addEventListener('timeupdate', () => updateByTime(video.currentTime));
  video.addEventListener('ended', () => updateByTime(0)); // snap on wrap

  // click-to-seek
  svg.addEventListener('click', (e) => {
    const rect = svg.getBoundingClientRect();
    const xRatio = (e.clientX - rect.left) / rect.width;
    const t = xRatio * video.duration;
    video.currentTime = Math.max(0, Math.min(video.duration, t));
    updateByTime(video.currentTime);
  });

  function make(tag, attrs){
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  updateByTime(0);
}
const io = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const v = entry.target;
    if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
      hydrateVideo(v);
      v.play().catch(()=>{});
    } else {
      v.pause();
      // לשחרור משאבים אם יש הרבה וידאו:
      // v.removeAttribute('src'); v.load();
    }
  });
}, { threshold: 0.5 });

// הרישום לצופה—רק על וידאו שלא בתוך קרוסלה,
// או פשוט על כולם אם נוח לך:
document.querySelectorAll('video').forEach(v => io.observe(v));