import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class MLP {

    private int[] tamanhos;
    private double[][][] pesos;
    private double[][] vieses;

    // Arrays pré-alocados para evitar Garbage Collection durante o treino
    private double[][] ativacoes;
    private double[][] deltas;

    private Random random = new Random(123);

    public MLP(int[] tamanhos) {
        this.tamanhos = tamanhos;
        int numCamadas = tamanhos.length;

        pesos = new double[numCamadas - 1][][];
        vieses = new double[numCamadas - 1][];

        // Inicializa os arrays de memória estática
        ativacoes = new double[numCamadas][];
        for (int i = 0; i < numCamadas; i++) {
            ativacoes[i] = new double[tamanhos[i]];
        }

        deltas = new double[numCamadas - 1][];
        for (int i = 0; i < numCamadas - 1; i++) {
            deltas[i] = new double[tamanhos[i + 1]];
        }

        double aleatorio = 0.2;
        for (int i = 0; i < numCamadas - 1; i++) {
            int neuroniosEntrada = tamanhos[i];
            int neuroniosSaida = tamanhos[i + 1];
            pesos[i] = new double[neuroniosSaida][neuroniosEntrada];
            vieses[i] = new double[neuroniosSaida];

            for (int j = 0; j < neuroniosSaida; j++) {
                vieses[i][j] = (random.nextDouble() * 2 * aleatorio) - aleatorio;
                for (int k = 0; k < neuroniosEntrada; k++) {
                    pesos[i][j][k] = (random.nextDouble() * 2 * aleatorio) - aleatorio;
                }
            }
        }
    }

    private double tanh(double x) {
        return Math.tanh(x);
    }

    private double derivadaTanh(double y) {
        return (1.0 + y) * (1.0 - y);
    }

    public double treinar(double[] entrada, double[] alvo, double taxaAprendizado) {
        int numCamadas = tamanhos.length;

        System.arraycopy(entrada, 0, ativacoes[0], 0, entrada.length);

        // --- FORWARD PROPAGATION ---
        for (int i = 0; i < numCamadas - 1; i++) {
            for (int j = 0; j < tamanhos[i + 1]; j++) {
                double soma = vieses[i][j];
                for (int k = 0; k < tamanhos[i]; k++) {
                    soma += pesos[i][j][k] * ativacoes[i][k];
                }
                ativacoes[i + 1][j] = tanh(soma);
            }
        }

        // --- CÁLCULO DO ERRO E BACKPROPAGATION ---
        double erroAmostra = 0.0;

        for (int j = 0; j < tamanhos[numCamadas - 1]; j++) {
            double y = ativacoes[numCamadas - 1][j];
            double diferenca = alvo[j] - y;
            erroAmostra += 0.5 * diferenca * diferenca;
            deltas[numCamadas - 2][j] = diferenca * derivadaTanh(y);
        }

        for (int i = numCamadas - 3; i >= 0; i--) {
            for (int j = 0; j < tamanhos[i + 1]; j++) {
                double erro = 0.0;
                for (int k = 0; k < tamanhos[i + 2]; k++) {
                    erro += pesos[i + 1][k][j] * deltas[i + 1][k];
                }
                double y = ativacoes[i + 1][j];
                deltas[i][j] = erro * derivadaTanh(y);
            }
        }

        // --- ATUALIZAÇÃO DE PESOS ---
        for (int i = 0; i < numCamadas - 1; i++) {
            for (int j = 0; j < tamanhos[i + 1]; j++) {
                vieses[i][j] += taxaAprendizado * deltas[i][j];
                for (int k = 0; k < tamanhos[i]; k++) {
                    pesos[i][j][k] += taxaAprendizado * deltas[i][j] * ativacoes[i][k];
                }
            }
        }

        return erroAmostra;
    }

    public int prever(double[] entrada) {
        System.arraycopy(entrada, 0, ativacoes[0], 0, entrada.length);

        for (int i = 0; i < tamanhos.length - 1; i++) {
            for (int j = 0; j < tamanhos[i + 1]; j++) {
                double soma = vieses[i][j];
                for (int k = 0; k < tamanhos[i]; k++) {
                    soma += pesos[i][j][k] * ativacoes[i][k];
                }
                ativacoes[i + 1][j] = tanh(soma);
            }
        }

        double menorDistancia = Double.MAX_VALUE;
        int classePrevista = -1;
        int ultimaCamada = tamanhos.length - 1;
        int numClasses = tamanhos[ultimaCamada]; // Dinâmico (26 para o alfabeto)

        for (int classe = 0; classe < numClasses; classe++) {
            double distanciaQuadrada = 0.0;
            for (int i = 0; i < numClasses; i++) {
                double valorIdeal = (i == classe) ? 1.0 : -1.0;
                double diff = valorIdeal - ativacoes[ultimaCamada][i];
                distanciaQuadrada += (diff * diff);
            }
            double distancia = Math.sqrt(distanciaQuadrada);

            if (distancia < menorDistancia) {
                menorDistancia = distancia;
                classePrevista = classe;
            }
        }

        return classePrevista;
    }

    public static double[][] carregarImagens(String caminho) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(caminho)));
        dis.readInt();
        int numImagens = dis.readInt();
        int numLinhas = dis.readInt();
        int numColunas = dis.readInt();
        int totalPixeis = numLinhas * numColunas;

        double[][] imagens = new double[numImagens][totalPixeis];

        byte[] buffer = new byte[numImagens * totalPixeis];
        dis.readFully(buffer);

        int indice = 0;
        for (int i = 0; i < numImagens; i++) {
            for (int j = 0; j < totalPixeis; j++) {
                imagens[i][j] = (buffer[indice++] & 0xFF) / 255.0;
            }
        }
        dis.close();
        return imagens;
    }

    public static int[] carregarLabels(String caminho) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(caminho)));
        dis.readInt();
        int numLabels = dis.readInt();

        int[] labels = new int[numLabels];
        byte[] buffer = new byte[numLabels];
        dis.readFully(buffer);

        for (int i = 0; i < numLabels; i++) {
            labels[i] = buffer[i] & 0xFF;
        }
        dis.close();
        return labels;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== MLP EMNIST (Letras A-Z) ===");

        String arquivoImagens = "imagens";
        String arquivoLabels = "labels";
        File fImagens = new File(arquivoImagens);
        File fLabels = new File(arquivoLabels);

        if (!fImagens.exists() || !fLabels.exists()) {
            System.out.println("\n[ERRO] Ficheiros de treino não encontrados na raiz do projeto!");
            return;
        }

        int numCamadas = -1;
        while (numCamadas < 0) {
            System.out.print("Quantas camadas escondidas? ");
            numCamadas = scanner.nextInt();
        }

        List<Integer> estrutura = new ArrayList<>();
        estrutura.add(784); // 28x28 pixels

        for (int i = 0; i < numCamadas; i++) {
            System.out.print("Quantos neurónios na camada " + (i + 1) + "? ");
            estrutura.add(scanner.nextInt());
        }
        estrutura.add(26); //26 (Letras A-Z)

        int[] tamanhos = estrutura.stream().mapToInt(i -> i).toArray();
        int numClasses = tamanhos[tamanhos.length - 1];
        MLP rede = new MLP(tamanhos);

        try {
            System.out.println("\nA carregar dataset de treino do EMNIST...");
            long inicioCarregamento = System.currentTimeMillis();
            double[][] imagensTreino = carregarImagens(arquivoImagens);
            int[] labelsTreino = carregarLabels(arquivoLabels);
            long fimCarregamento = System.currentTimeMillis();
            System.out.println("Dataset carregado em: " + (fimCarregamento - inicioCarregamento) + " ms");

            int numAmostras = 10000;
            double taxaAprendizado = 0.003;
            double acuraciaAlvo = 88.0; 

            System.out.println("\nIniciando o treino.\nNumero de Amostras: " + numAmostras + "\nMeta: Parar ao atingir "
                    + acuraciaAlvo + "% de acertos.");
            System.out.println("Ciclo\t\tErro Total\t\tAcurácia\t\tTempo(ms)");
            System.out.println("-------------------------------------------------------------------------");

            int epoca = 1;
            double acuraciaAtual = 0.0;

            while (acuraciaAtual < acuraciaAlvo) {
                long inicioEpoca = System.currentTimeMillis();
                int acertos = 0;
                double erroTotalDaEpoca = 0.0;

                for (int i = 0; i < numAmostras; i++) {
                    double[] imagem = imagensTreino[i];
                    
                    // O EMNIST de letras costuma vir indexado de 1 a 26. Subtraímos 1 para ir de 0 a 25.
                    int labelReal = labelsTreino[i] - 1; 
                    if (labelReal < 0) labelReal = 0;

                    double[] alvo = new double[numClasses];
                    for (int j = 0; j < numClasses; j++)
                        alvo[j] = -1.0;
                    alvo[labelReal] = 1.0;

                    int previsao = rede.prever(imagem);
                    if (previsao == labelReal)
                        acertos++;

                    erroTotalDaEpoca += rede.treinar(imagem, alvo, taxaAprendizado);
                }

                long tempoEpoca = System.currentTimeMillis() - inicioEpoca;
                acuraciaAtual = (acertos / (double) numAmostras) * 100;
                System.out.printf("%d\t\t%.4f\t\t%.2f%%\t\t\t%d ms\n", epoca, erroTotalDaEpoca, acuraciaAtual,
                        tempoEpoca);
                epoca++;
            }

            System.out.println("-------------------------------------------------------------------------");
            System.out.println("Treino finalizado! Meta alcançada.\n");
            System.out.printf("Qtd de ciclos: %d\n", epoca - 1);

            // --- FASE DE TESTE ---
            System.out.println("\n==================================================");
            System.out.println("INICIANDO A FASE DE TESTE COM O DATASET DE TESTE");
            System.out.println("==================================================");
            String arquivoImagensTeste = "imagens_teste";
            String arquivoLabelsTeste = "labels_teste";
            File fImagensTeste = new File(arquivoImagensTeste);
            File fLabelsTeste = new File(arquivoLabelsTeste);

            if (fImagensTeste.exists() && fLabelsTeste.exists()) {
                System.out.println("A carregar os ficheiros de teste...\n");
                double[][] imagensTeste = carregarImagens(arquivoImagensTeste);
                int[] labelsTeste = carregarLabels(arquivoLabelsTeste);
                int numAmostrasTeste = 2000;
                int acertosTeste = 0;

                System.out.println("Amostra\t\tPrevisto\tReal\t\tStatus");
                System.out.println("--------------------------------------------------");

                for (int i = 0; i < numAmostrasTeste; i++) {
                    int previsao = rede.prever(imagensTeste[i]);
                    int real = labelsTeste[i] - 1; // Ajuste do índice EMNIST (1-26 para 0-25)
                    if (real < 0) real = 0;

                    boolean acertou = (previsao == real);
                    if (acertou) {
                        acertosTeste++;
                    }

                    if ((0 <= i && i < 5) || (1000 <= i && i < 1005) || (3000 <= i && i < 3005) ) {
                        char letraPrevista = (char) ('A' + previsao);
                        char letraReal = (char) ('A' + real);
                        String status = acertou ? "[ACERTO]" : "[ERRO]";
                        System.out.printf("Teste #%d\t\t  %c\t\t  %c\t\t%s\n", (i + 1), letraPrevista, letraReal, status);
                    }
                }

                System.out.println("--------------------------------------------------");
                System.out.println("[As amostras restantes foram processadas internamente...]\n");

                int errosTeste = numAmostrasTeste - acertosTeste;
                double acuraciaTeste = (acertosTeste / (double) numAmostrasTeste) * 100;

                System.out.println("==================================================");
                System.out.println("ESTATÍSTICAS FINAIS DA MLP (LETRAS):");
                System.out.println("Tamanho da amostra de treino: " + numAmostras);
                System.out.println("Meta de treino: " + acuraciaAlvo + "%");
                System.out.println("Acurácia atingida: " + acuraciaAtual + "%");
                System.out.println("Total de imagens testadas: " + numAmostrasTeste);
                System.out.println("Total de acertos: " + acertosTeste);
                System.out.println("Total de erros: " + errosTeste);
                System.out.printf("Taxa de acurácia nos testes: %.2f%%\n", acuraciaTeste);
                System.out.println("==================================================");

            } else {
                System.out.println("\n[AVISO] Ficheiros 'imagens_teste' e 'labels_teste' não encontrados.");
            }
        } catch (IOException e) {
            System.out.println("Erro ao ler os ficheiros: " + e.getMessage());
        }
        scanner.close();
    }
}