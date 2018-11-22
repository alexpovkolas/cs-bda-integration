Разработать MPI-приложение для расчета интеграла методом трапеции на заданном количестве процессов. Интервал интегрирования разбивается на n отрезков, затем множество отрезков последовательно разбивается на блоки по r отрезков, количество блоков должно быть больше количества процессов p. Далее необходимо реализовать динамическое распределение вычислительной нагрузки в рамках “master-slave” модели: мастер-процесс (0-процесс) по запросу распределяет по одному блоку между ненулевыми процессами (на старте можно без запроса), рабочий процесс получает задание-блок, расчитывает частичную сумму и делает запрос на следующий блок. Частичные суммы процесс может накапливать локально, передавая результат мастер-процессу в конце обработки своего набора блоков.
Примечания:
Входные параметры задачи: количество процессов p, количество отрезков разбиения интервала интегрирования n, размер блока отрезков r для динамического распределения вычислительной нагрузки.
