// API endpoint'leri
const API_BASE = window.location.origin;

// Sayfa yüklendiğinde veri setlerini getir ve arayüzü ayarla
document.addEventListener('DOMContentLoaded', async () => {
    const datasetSelect = document.getElementById('datasetSelect');
    const textAnalysisSection = document.querySelector('.text-analysis');
    const imageAnalysisSection = document.querySelector('.image-analysis');
    const textResultBox = document.getElementById('textResult');
    const imageResultBox = document.getElementById('imageResult');

    // Başlangıçta analiz bölümlerini gizle
    textAnalysisSection.style.display = 'none';
    imageAnalysisSection.style.display = 'none';

    // Veri seti seçimi değiştiğinde
    datasetSelect.addEventListener('change', () => {
        const selectedOption = datasetSelect.options[datasetSelect.selectedIndex];
        const selectedDatasetName = selectedOption.value;
        
        // 'Veri seti seçiniz...' seçeneği seçildiğinde her şeyi gizle
        if (!selectedDatasetName) {
            textAnalysisSection.style.display = 'none';
            imageAnalysisSection.style.display = 'none';
            document.getElementById('textInput').value = '';
            textResultBox.innerHTML = '';
            document.getElementById('imageInput').value = '';
            imageResultBox.innerHTML = '';
            showInfo('Lütfen analiz yapmak için bir veri seti seçin.');
            return;
        }

        const supportedTypes = selectedOption.dataset.types ? JSON.parse(selectedOption.dataset.types) : [];

        // Analiz bölümlerini seçilen veri setinin desteklediği türlere göre göster/gizle
        if (supportedTypes.includes('text')) {
            textAnalysisSection.style.display = 'block';
        } else {
            textAnalysisSection.style.display = 'none';
            // Metin alanını ve sonucu temizle
            document.getElementById('textInput').value = '';
            textResultBox.innerHTML = '';
        }

        if (supportedTypes.includes('image')) {
            imageAnalysisSection.style.display = 'block';
        } else {
            imageAnalysisSection.style.display = 'none';
            // Görsel alanını ve sonucu temizle
            document.getElementById('imageInput').value = '';
            imageResultBox.innerHTML = '';
        }

        showInfo(`"${selectedDatasetName}" veri seti seçildi. Analiz yapmaya hazırsınız.`);
    });
    
    // Dosya seçimi değiştiğinde (existing logic remains)
    const imageInput = document.getElementById('imageInput');
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (!file.type.startsWith('image/')) {
                showError('Lütfen geçerli bir görsel dosyası seçin.');
                imageInput.value = ''; // Clear the file input
                return;
            }
            showInfo(`"${file.name}" görseli seçildi.`);
        }
    });

    
    // Veri setlerini yükle
    try {
        const res = await fetch(`${API_BASE}/datasets`);
        if (!res.ok) {
             const errorDetail = await res.text().catch(() => res.statusText);
             throw new Error(`Veri setleri yüklenemedi: ${res.status} - ${errorDetail}`);
        }
        
        const datasets = await res.json();
        const select = document.getElementById("datasetSelect");
        
        // Mevcut seçenekleri temizle (ilk "Veri seti seçiniz..." hariç)
        select.innerHTML = '<option value="">Veri seti seçiniz...</option>';

        if (datasets.length === 0) {
            showError('Backend\'den yüklenecek veri seti bulunamadı.');
            return;
        }

        datasets.forEach(ds => {
            const opt = document.createElement("option");
            opt.value = ds.name;
            let textContent = ds.name;

            if (ds.types && ds.types.length > 0) {
                const typeLabels = [];
                if (ds.types.includes('text')) typeLabels.push('Metin');
                if (ds.types.includes('image')) typeLabels.push('Görsel');
                textContent += ` (${typeLabels.join(', ')})`;
            } else if (ds.error) {
                 textContent += ` (Hata)`; // Indicate error if types are missing and error exists
            }

            opt.textContent = textContent;
            // Desteklenen tür bilgisini data attribute olarak ekle (dinamik gösterme/gizleme ve doğrulama için)
            opt.dataset.types = JSON.stringify(ds.types || []); // Ensure types is always an array in dataset
            // Hata bilgisi varsa data attribute olarak ekle
            if(ds.error) {
                opt.dataset.error = ds.error;
            }
            select.appendChild(opt);
        });

         // Sayfa yüklendikten sonra ilk seçeneği seçili hale getir ve change eventini tetikle
        datasetSelect.selectedIndex = 0; // Select 'Veri seti seçiniz...' option
        datasetSelect.dispatchEvent(new Event('change'));


    } catch (error) {
        showError('Veri setleri yüklenirken bir hata oluştu: ' + error.message);
    }
});

// Metin analizi fonksiyonu
async function predictText() {
    const datasetSelect = document.getElementById("datasetSelect");
    const dataset = datasetSelect.value;
    const selectedOption = datasetSelect.options[datasetSelect.selectedIndex];
    const supportedTypes = selectedOption.dataset.types ? JSON.parse(selectedOption.dataset.types) : [];

    // Frontend doğrulaması: Seçili veri seti metin analizini destekliyor mu?
    if (!supportedTypes.includes('text')) {
         showError('Seçilen veri seti metin analizi için uygun değil.', document.getElementById('textResult'));
         return;
    }

    const text = document.getElementById("textInput").value;
    const resultBox = document.getElementById("textResult");

    if (!dataset) {
        showError('Lütfen bir veri seti seçin', resultBox);
        return;
    }
    if (!text.trim()) {
        showError('Lütfen analiz edilecek metni girin', resultBox);
        return;
    }

    try {
        resultBox.innerHTML = '<div class="loading">Metin analiz ediliyor</div>';
        
        const res = await fetch(`${API_BASE}/predict_text`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({dataset, text})
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => null);
            const errorMessage = errorData && errorData.detail ? errorData.detail : 'Analiz yapılamadı';
            throw new Error(`API Hatası: ${errorMessage}`);
        }
        
        const data = await res.json();
        resultBox.innerHTML = `
            <div class="success">
                <strong>Analiz Sonucu:</strong><br>
                Sınıf: ${data.label}<br>
                Kod: ${data.code}
            </div>`;
    } catch (error) {
        showError('Metin analizi sırasında bir hata oluştu: ' + error.message, resultBox);
    }
}

// Görsel analizi fonksiyonu
async function predictImage() {
    const datasetSelect = document.getElementById("datasetSelect");
    const dataset = datasetSelect.value;
     const selectedOption = datasetSelect.options[datasetSelect.selectedIndex];
    const supportedTypes = selectedOption.dataset.types ? JSON.parse(selectedOption.dataset.types) : [];

     // Frontend doğrulaması: Seçili veri seti görsel analizini destekliyor mu?
    if (!supportedTypes.includes('image')) {
         showError('Seçilen veri seti görsel analizi için uygun değil.', document.getElementById('imageResult'));
         return;
    }

    const file = document.getElementById("imageInput").files[0];
    const resultBox = document.getElementById("imageResult");

    if (!dataset) {
        showError('Lütfen bir veri seti seçin', resultBox);
        return;
    }
    if (!file) {
        showError('Lütfen bir görsel seçin', resultBox);
        return;
    }

    try {
        resultBox.innerHTML = '<div class="loading">Görsel analiz ediliyor</div>';
        
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${API_BASE}/predict_image?dataset=${dataset}`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => null);
            const errorMessage = errorData && errorData.detail ? errorData.detail : 'Analiz yapılamadı';
            throw new Error(`API Hatası: ${errorMessage}`);
        }
        
        const data = await res.json();
        resultBox.innerHTML = `
            <div class="success">
                <strong>Analiz Sonucu:</strong><br>
                Sınıf: ${data.label}<br>
                Kod: ${data.code}
            </div>`;
    } catch (error) {
        showError('Görsel analizi sırasında bir hata oluştu: ' + error.message, resultBox);
    }
}

// Hata mesajı gösterme fonksiyonu
function showError(message, element = null) {
    // Önceki hata/bilgi mesajlarını temizle
    document.querySelectorAll('.error, .info').forEach(el => el.remove());

    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    
    if (element) {
        element.innerHTML = '';
        element.appendChild(errorDiv);
    } else {
        // Genel hata mesajı (üstte çıkar)
        document.body.insertBefore(errorDiv, document.body.firstChild);
        // Otomatik kapanma
        setTimeout(() => errorDiv.remove(), 7000); // Hata mesajları daha uzun kalsın
    }
}

// Bilgi mesajı gösterme fonksiyonu
function showInfo(message) {
     // Önceki hata/bilgi mesajlarını temizle
     document.querySelectorAll('.error, .info').forEach(el => el.remove());

    const infoDiv = document.createElement('div');
    infoDiv.className = 'info';
    infoDiv.textContent = message;
    
    document.body.insertBefore(infoDiv, document.body.firstChild);
    setTimeout(() => infoDiv.remove(), 5000);
}
